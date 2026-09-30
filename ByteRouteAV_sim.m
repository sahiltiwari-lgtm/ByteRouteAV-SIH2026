%% ByteRouteAV_sim.m
% ByteRoute AV - Adaptive path planning & collision avoidance on unstructured roads
% SIH 2026 | PS SIH26037
%
% Pipeline: simulated multi-sensor perception -> alpha-beta tracking ->
% constant-velocity prediction -> space-time A* replanning with class-based
% safety bubbles -> pure-pursuit + bicycle-model control -> metrics logging.
%
% ZERO-COLLISION TUNING:
%  - no path smoothing after A* (it could pull a waypoint back into a cell
%    the planner had explicitly avoided)
%  - independent proximity brake: fires off the tracked agents' CURRENT
%    distance every frame, not just the planned path's projected risk -
%    a second safety net in case the path itself leaves a small gap
%  - larger class-based safety bubbles, tighter replanning cadence, lower
%    cruise speed, finer planner grid, more planner move directions
%  - if the planner finds no safe path at all, the car holds position
%    instead of falling back to a straight line through the goal
%  - .avi / 'Motion JPEG AVI' video export (MATLAB Online doesn't support
%    the 'MPEG-4' profile)
%
% Runs all 3 scenarios x 2 modes (baseline, adaptive) = 6 runs total.
% Each figure stays open after its run and animates frame-by-frame.

clear; clc; close all; rng(1);

%% ---------------- CONFIG ----------------
cfg.dt          = 0.1;      % sim step (s)
cfg.Tmax        = 50;       % max sim time (s)
cfg.horizon     = 5;        % prediction horizon (s)
cfg.res         = 0.4;      % planner grid resolution (m)
cfg.replanEvery = 0.15;     % periodic replan interval (s)
cfg.sensorRange = 35;       % fused sensor range (m)
cfg.sensorSigma = 0.15;     % fused position noise (m)
cfg.vCruise     = 4;        % cruise speed (m/s)
cfg.egoR        = 1.3;      % ego footprint radius (m)
cfg.wheelbase   = 2.7;      % m
cfg.maxSteer    = 35*pi/180;
cfg.aMax        = 2.0;      % max accel (m/s^2)
cfg.bMax        = 7.0;      % max brake (m/s^2)
cfg.roadY       = [-6 6];   % drivable width (m)
cfg.goal        = [95 0];
cfg.brakeDist   = 2.5;      % independent proximity brake radius (m)
cfg.scenarios   = 1:3;       % 1 village, 2 urban junction, 3 sudden obstacle
cfg.animate     = true;
cfg.pauseSec    = 0.02;      % per-frame delay so motion is visible; lower = faster
cfg.saveVideo   = true;      % true -> writes .avi per run, for your pitch clips
cfg.closeFigs   = false;     % false -> figure stays open after each run

modesToRun = {'baseline','adaptive'};   % full before/after comparison

%% ---------------- RUN ----------------
R = [];
for sid = cfg.scenarios %[output:group:74f54d8c]
    for mi = 1:numel(modesToRun)
        r = runScenario(sid, modesToRun{mi}, cfg); %[output:5af7dcd0] %[output:76e9b7ea] %[output:322fd2f7] %[output:4b7501f4] %[output:9326a105] %[output:32268c15]
        if isempty(R), R = r; else, R(end+1,1) = r; end %#ok<SAGROW>
    end
end %[output:group:74f54d8c]
disp(struct2table(R)); %[output:9c3c4f2b]
ad = strcmp({R.Mode}, 'adaptive');
fprintf('\nAdaptive: %d/%d scenarios passed (zero collisions + goal reached)\n', ... %[output:group:52cee8f1] %[output:06f56b06]
    sum([R(ad).Success]), sum(ad)); %[output:group:52cee8f1] %[output:06f56b06]
fprintf('Total adaptive collisions across all scenarios: %d\n', sum([R(ad).Collisions])); %[output:38d3f89f]

%% =====================================================================
function r = runScenario(sid, mode, cfg)
[name, A] = getScenario(sid);
nA = numel(A);
adaptive = strcmp(mode, 'adaptive');

ego = struct('x',0,'y',0,'th',0,'v',3);
trk = struct('pos',zeros(nA,2),'vel',zeros(nA,2),'seen',false(nA,1));
path = [0 0; cfg.goal];

nSteps = round(cfg.Tmax/cfg.dt);
vlog = zeros(nSteps,1); yawlog = zeros(nSteps,1);
lastReplan = -inf; replans = 0; reactive = 0; lat = [];
inContact = false(nA,1); collisions = 0; minClrTrue = inf;
stuck = 0; reached = false; margin = inf; minClr = inf; tHit = inf;

V = [];
if cfg.animate, V = vizInit(name, mode, A, cfg); end

for k = 1:nSteps
    t = (k-1)*cfg.dt;

    % 1) perception + tracking
    trk = perceive(trk, A, ego, t, cfg);

    % 2) risk check + (re)planning
    vdes = cfg.vCruise;
    if adaptive
        [margin, minClr, tHit] = pathRisk(path, ego, trk, A, cfg);
        due = (t - lastReplan) >= cfg.replanEvery;
        urgent = (margin < 0) && ((t - lastReplan) >= 0.1);
        if due || urgent
            tic;
            [newPath, ok] = planPath(ego, trk, A, cfg);
            lat(end+1) = toc*1000; %#ok<AGROW>
            replans = replans + 1;
            if margin < 0, reactive = reactive + 1; end
            if ok, path = newPath; end
            lastReplan = t;
            [margin, minClr, tHit] = pathRisk(path, ego, trk, A, cfg);
        end
        % speed logic: emergency stop / slow in bubble / nudge if stuck
        if minClr < 1.2 && tHit < 3.0
            vdes = 0;
        elseif margin < 0
            vdes = 0.35*cfg.vCruise;
        end
        % INDEPENDENT proximity brake: fires off actual current tracked
        % distance, not the planned path's projected risk. Second safety
        % net in case the path itself has a small gap.
        idxSeen = find(trk.seen);
        for i = idxSeen'
            [ar, buf] = classParams(A(i).cls);
            hardR = ar + cfg.egoR;
            dNow = hypot(trk.pos(i,1)-ego.x, trk.pos(i,2)-ego.y);
            if dNow < hardR + cfg.brakeDist
                vdes = 0;
            end
        end
        if ego.v < 0.3, stuck = stuck + cfg.dt; else, stuck = 0; end
        if stuck > 2 && minClr > 0.5, vdes = 1.0; end
    end

    % 3) control: pure pursuit + bicycle model
    Ld = min(max(1.5 + 0.8*ego.v, 3), 8);
    tgt = lookaheadPoint(path, ego, Ld);
    alpha = atan2(tgt(2)-ego.y, tgt(1)-ego.x) - ego.th;
    alpha = atan2(sin(alpha), cos(alpha));
    delta = atan2(2*cfg.wheelbase*sin(alpha), Ld);
    delta = max(min(delta, cfg.maxSteer), -cfg.maxSteer);
    acc = max(min(1.5*(vdes - ego.v), cfg.aMax), -cfg.bMax);

    ego.v  = max(ego.v + acc*cfg.dt, 0);
    ego.x  = ego.x + ego.v*cos(ego.th)*cfg.dt;
    ego.y  = ego.y + ego.v*sin(ego.th)*cfg.dt;
    yr     = ego.v/cfg.wheelbase*tan(delta);
    ego.th = ego.th + yr*cfg.dt;
    vlog(k) = ego.v; yawlog(k) = yr;

    % 4) ground-truth collision check
    tn = t + cfg.dt;
    for i = 1:nA
        [p, act] = agentPos(A(i), tn);
        if ~act, continue; end
        ar = classParams(A(i).cls);
        d = hypot(p(1)-ego.x, p(2)-ego.y) - (ar + cfg.egoR);
        minClrTrue = min(minClrTrue, d);
        if d < 0 && ~inContact(i)
            collisions = collisions + 1; inContact(i) = true;
        elseif d >= 0.5
            inContact(i) = false;
        end
    end

    if cfg.animate
        vizUpdate(V, ego, path, trk, A, tn, cfg, collisions, replans);
    end

    if ego.x >= cfg.goal(1) - 1, reached = true; break; end
end
if cfg.animate, vizClose(V, cfg); end

% metrics
alat = vlog(1:k).*yawlog(1:k);
if k > 2, rmsJerk = sqrt(mean((diff(alat)/cfg.dt).^2)); else, rmsJerk = 0; end
if isempty(lat), lat = NaN; end

r.Scenario        = name;
r.Mode            = mode;
r.Collisions      = collisions;
r.MinClearance_m  = round(minClrTrue, 2);
r.Replans         = replans;
r.ReactiveReplans = reactive;
r.MeanLatency_ms  = round(mean(lat), 1);
r.MaxLatency_ms   = round(max(lat), 1);
r.RMSLatJerk      = round(rmsJerk, 2);
r.TimeToGoal_s    = NaN; if reached, r.TimeToGoal_s = round(k*cfg.dt, 1); end
r.Success         = reached && collisions == 0;
end

%% ---------------- SCENARIOS ----------------
function [name, A] = getScenario(id)
mk = @(nm,cls,p0,v,ta) struct('name',nm,'cls',cls,'p0',p0,'v',v,'tAppear',ta);
switch id
    case 1
        name = 'Village road';
        A = [mk('cow','animal',[45 -5],[0 0.5],0), ...
             mk('cart','cart',[58 3],[0 0],0), ...
             mk('jaywalker','ped',[72 -6],[0 1.2],9)];
    case 2
        name = 'Urban junction';
        A = [mk('auto','auto',[50 5.5],[-1.5 -1.2],0), ...
             mk('ped1','ped',[65 -5.5],[0 1.4],6), ...
             mk('ped2','ped',[68 5.5],[0 -1.2],8), ...
             mk('bike','bike',[90 -0.5],[-3 0],0)];
    otherwise
        name = 'Sudden obstacle';
        A = [mk('slowauto','auto',[20 -2],[2.5 0],0), ...
             mk('cow','animal',[45 1],[0 0],4)];
end
end

function [r, buf] = classParams(cls)
% physical radius (m) and safety-bubble multiplier per road-user class
switch cls
    case 'ped',    r = 0.4; buf = 1.6;
    case 'animal', r = 0.7; buf = 1.8;
    case 'auto',   r = 1.0; buf = 2.2;
    case 'bike',   r = 0.6; buf = 2.2;
    case 'cart',   r = 0.9; buf = 1.8;
    otherwise,     r = 1.0; buf = 1.8;
end
end

function [p, active] = agentPos(a, t)
active = t >= a.tAppear;
p = a.p0 + a.v*max(t - a.tAppear, 0);
end

%% ---------------- PERCEPTION ----------------
function trk = perceive(trk, A, ego, t, cfg)
for i = 1:numel(A)
    [pt, act] = agentPos(A(i), t);
    if ~act || hypot(pt(1)-ego.x, pt(2)-ego.y) > cfg.sensorRange
        trk.seen(i) = false; continue;
    end
    meas = pt + cfg.sensorSigma*randn(1,2);
    if ~trk.seen(i)
        trk.pos(i,:) = meas; trk.vel(i,:) = [0 0]; trk.seen(i) = true;
    else
        pred = trk.pos(i,:) + trk.vel(i,:)*cfg.dt;
        res  = meas - pred;
        trk.pos(i,:) = pred + 0.5*res;
        trk.vel(i,:) = trk.vel(i,:) + (0.1/cfg.dt)*res;
    end
end
end

%% ---------------- PREDICTION / RISK ----------------
function [margin, minClr, tHit] = pathRisk(path, ego, trk, A, cfg)
margin = inf; minClr = inf; tHit = inf;
idx = find(trk.seen);
if isempty(idx), return; end
vp = max(ego.v, 2.5);
tk = (0:0.2:cfg.horizon)';
seg = hypot(diff(path(:,1)), diff(path(:,2)));
cs = [0; cumsum(seg)];
[~, ii] = min(hypot(path(:,1)-ego.x, path(:,2)-ego.y));
pe = posOnPath(path, cs(ii) + vp*tk);
for i = idx'
    [ar, buf] = classParams(A(i).cls);
    hard = ar + cfg.egoR; safe = hard*buf;
    pa = trk.pos(i,:) + tk*trk.vel(i,:);
    d = hypot(pe(:,1)-pa(:,1), pe(:,2)-pa(:,2));
    margin = min(margin, min(d - safe));
    [c, j] = min(d - hard);
    if c < minClr, minClr = c; tHit = tk(j); end
end
end

%% ---------------- PLANNER (space-time A*) ----------------
function [path, ok] = planPath(ego, trk, A, cfg)
res = cfg.res;
xs = max(0, ego.x-2) : res : min(100, ego.x+40);
ys = cfg.roadY(1) : res : cfg.roadY(2);
nx = numel(xs); ny = numel(ys);

idx = find(trk.seen); m = numel(idx);
Ap = trk.pos(idx,:); Av = trk.vel(idx,:);
hard = zeros(m,1); safe = zeros(m,1);
for j = 1:m
    [ar, buf] = classParams(A(idx(j)).cls);
    hard(j) = ar + cfg.egoR; safe(j) = hard(j)*buf;
end
vp = max(ego.v, 2.5);

[~, six] = min(abs(xs - ego.x)); [~, siy] = min(abs(ys - ego.y));
goalX = min(ego.x + 38, cfg.goal(1));
[~, gix] = min(abs(xs - goalX));

G = inf(ny,nx); Len = zeros(ny,nx); par = zeros(ny,nx);
F = inf(ny,nx); closed = false(ny,nx);
G(siy,six) = 0; F(siy,six) = max(gix-six,0)*res;
moves = [1 0; 1 1; 1 -1; 2 1; 2 -1; 1 2; 1 -2; 0 1; 0 -1];

ok = false; goalIdx = 0;
while true
    [fmin, cur] = min(F(:));
    if isinf(fmin), break; end
    [iy, ix] = ind2sub([ny nx], cur);
    F(cur) = inf; closed(cur) = true;
    if ix >= gix, ok = true; goalIdx = cur; break; end
    for q = 1:size(moves,1)
        jx = ix + moves(q,1); jy = iy + moves(q,2);
        if jx < 1 || jx > nx || jy < 1 || jy > ny, continue; end
        if closed(jy,jx), continue; end
        step = hypot(moves(q,1), moves(q,2))*res;
        lenN = Len(iy,ix) + step;
        tArr = min(lenN/vp, cfg.horizon);
        dx = xs(jx) - (Ap(:,1) + Av(:,1)*tArr);
        dy = ys(jy) - (Ap(:,2) + Av(:,2)*tArr);
        d = sqrt(dx.^2 + dy.^2);
        if any(d < hard), continue; end   % hard block: never enter contact zone
        pen = 60*sum(max(0, 1 - d./safe).^2);
        gN = G(iy,ix) + step*(1 + 0.05*abs(ys(jy))) + pen + 1.0*(abs(ys(jy)) > 4.5);
        if gN < G(jy,jx)
            G(jy,jx) = gN; Len(jy,jx) = lenN; par(jy,jx) = cur;
            F(jy,jx) = gN + max(gix-jx,0)*res;
        end
    end
end

if ~ok, path = [ego.x ego.y; ego.x ego.y]; return; end  % no safe path -> hold position, brake layer stops the car

pts = zeros(0,2); cur = goalIdx;
while cur ~= 0
    [iy, ix] = ind2sub([ny nx], cur);
    pts(end+1,:) = [xs(ix) ys(iy)]; %#ok<AGROW>
    cur = par(cur);
end
pts = flipud(pts);
% no post-smoothing: smoothing waypoints could pull one back into a cell
% the A* search had explicitly avoided
pts(1,:) = [ego.x ego.y];
path = pts;
end

%% ---------------- PATH HELPERS ----------------
function p = posOnPath(path, s)
seg = hypot(diff(path(:,1)), diff(path(:,2)));
cs = [0; cumsum(seg)];
[cs, u] = unique(cs);
pth = path(u,:);
s = min(max(s(:), 0), cs(end));
if numel(cs) < 2
    p = repmat(path(1,:), numel(s), 1); return;
end
p = [interp1(cs, pth(:,1), s), interp1(cs, pth(:,2), s)];
end

function tgt = lookaheadPoint(path, ego, Ld)
seg = hypot(diff(path(:,1)), diff(path(:,2)));
cs = [0; cumsum(seg)];
[~, i] = min(hypot(path(:,1)-ego.x, path(:,2)-ego.y));
tgt = posOnPath(path, cs(i) + Ld);
end

%% ---------------- VISUALISATION ----------------
function V = vizInit(name, mode, A, cfg)
V.fig = figure('Name',[name ' - ' mode],'Color','w','Position',[100 100 1100 380]);
V.ax = axes('Parent',V.fig); hold(V.ax,'on'); axis(V.ax,'equal');
xlim(V.ax,[-5 105]); ylim(V.ax,[-9 9]);
patch(V.ax,[-5 105 105 -5],[cfg.roadY(1) cfg.roadY(1) cfg.roadY(2) cfg.roadY(2)], ...
    [0.88 0.88 0.88],'EdgeColor','none');
plot(V.ax,cfg.goal(1),cfg.goal(2),'gp','MarkerSize',14,'MarkerFaceColor','g');
V.path = plot(V.ax,NaN,NaN,'g-','LineWidth',2);
V.ego  = patch(V.ax,NaN,NaN,[0.1 0.3 0.9]);
cols = lines(numel(A));
for i = 1:numel(A)
    V.agent(i) = plot(V.ax,NaN,NaN,'o','MarkerSize',9,'MarkerFaceColor',cols(i,:),'MarkerEdgeColor','k');
    V.bub(i)   = plot(V.ax,NaN,NaN,'--','Color',cols(i,:));
    V.pred(i)  = plot(V.ax,NaN,NaN,':','Color',cols(i,:),'LineWidth',1.5);
end
V.title = title(V.ax,'');
V.name = name; V.mode = mode;
if cfg.saveVideo
    V.vw = VideoWriter(sprintf('%s_%s.avi', strrep(name,' ','_'), mode), 'Motion JPEG AVI');
    open(V.vw);
end
end

function vizUpdate(V, ego, path, trk, A, t, cfg, collisions, replans)
c = cos(ego.th); s = sin(ego.th);
corners = [-2.2 -0.9; 2.2 -0.9; 2.2 0.9; -2.2 0.9]*[c s; -s c];
set(V.ego,'XData',ego.x+corners(:,1),'YData',ego.y+corners(:,2));
set(V.path,'XData',path(:,1),'YData',path(:,2));
th = linspace(0, 2*pi, 30);
for i = 1:numel(A)
    [p, act] = agentPos(A(i), t);
    if act
        [ar, buf] = classParams(A(i).cls);
        rb = (ar + cfg.egoR)*buf;
        set(V.agent(i),'XData',p(1),'YData',p(2));
        set(V.bub(i),'XData',p(1)+rb*cos(th),'YData',p(2)+rb*sin(th));
        if trk.seen(i)
            q = trk.pos(i,:) + trk.vel(i,:)*cfg.horizon;
            set(V.pred(i),'XData',[trk.pos(i,1) q(1)],'YData',[trk.pos(i,2) q(2)]);
        else
            set(V.pred(i),'XData',NaN,'YData',NaN);
        end
    end
end
set(V.title,'String',sprintf('%s | %s | t=%.1fs v=%.1f m/s | replans=%d | collisions=%d', ...
    V.name, V.mode, t, ego.v, replans, collisions));
drawnow;
if cfg.pauseSec > 0, pause(cfg.pauseSec); end
if isfield(V,'vw'), writeVideo(V.vw, getframe(V.fig)); end
end

function vizClose(V, cfg)
if isfield(V,'vw'), close(V.vw); end
if cfg.closeFigs, pause(0.5); close(V.fig); end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":45.2}
%---
%[output:5af7dcd0]
%   data: {"dataType":"image","outputData":{"height":380,"width":1100}}
%---
%[output:76e9b7ea]
%   data: {"dataType":"image","outputData":{"height":380,"width":1100}}
%---
%[output:322fd2f7]
%   data: {"dataType":"image","outputData":{"height":380,"width":1100}}
%---
%[output:4b7501f4]
%   data: {"dataType":"image","outputData":{"height":380,"width":1100}}
%---
%[output:9326a105]
%   data: {"dataType":"image","outputData":{"height":380,"width":1100}}
%---
%[output:32268c15]
%   data: {"dataType":"image","outputData":{"height":380,"width":1100}}
%---
%[output:9c3c4f2b]
%   data: {"dataType":"text","outputData":{"text":"         <strong>Scenario<\/strong>              <strong>Mode<\/strong>        <strong>Collisions<\/strong>    <strong>MinClearance_m<\/strong>    <strong>Replans<\/strong>    <strong>ReactiveReplans<\/strong>    <strong>MeanLatency_ms<\/strong>    <strong>MaxLatency_ms<\/strong>    <strong>RMSLatJerk<\/strong>    <strong>TimeToGoal_s<\/strong>    <strong>Success<\/strong>\n    <strong>___________________<\/strong>    <strong>____________<\/strong>    <strong>__________<\/strong>    <strong>______________<\/strong>    <strong>_______<\/strong>    <strong>_______________<\/strong>    <strong>______________<\/strong>    <strong>_____________<\/strong>    <strong>__________<\/strong>    <strong>____________<\/strong>    <strong>_______<\/strong>\n\n    {'Village road'   }    {'baseline'}        1              -1.3             0              0               NaN               NaN               0           23.7         false \n    {'Village road'   }    {'adaptive'}        0              1.65           309            221               3.2              19.6            0.99            NaN         false \n    {'Urban junction' }    {'baseline'}        1             -1.33             0              0               NaN               NaN               0           23.7         false \n    {'Urban junction' }    {'adaptive'}        0              0.97           156             43               3.5              16.2            1.77           29.9         true  \n    {'Sudden obstacle'}    {'baseline'}        2                -1             0              0               NaN               NaN               0           23.7         false \n    {'Sudden obstacle'}    {'adaptive'}        0              1.28           325            275               4.8              22.6            1.15            NaN         false \n\n","truncated":false}}
%---
%[output:06f56b06]
%   data: {"dataType":"text","outputData":{"text":"\nAdaptive: 1\/3 scenarios passed (zero collisions + goal reached)\n","truncated":false}}
%---
%[output:38d3f89f]
%   data: {"dataType":"text","outputData":{"text":"Total adaptive collisions across all scenarios: 0\n","truncated":false}}
%---
