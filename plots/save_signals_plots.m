%% ===============================================================
%  DOCUMENTED SIGNALS – ALL CUTTERS (c1–c6) – FINAL FIX
%% ===============================================================

clear; clc; close all;

DATA_ROOT = "E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling";
OUT_ROOT  = "E:\4 Paper\4 Paper\Results\final_final_diagrams\documented_signals";

FS = 50000;
cutters = ["c1","c2","c3","c4","c5","c6"];

sensor_idx = 1;        % Force X
SEG_TIME   = 0.2;
N          = round(SEG_TIME * FS);

% Colors (IEEE / MATLAB safe)
COL_TIME = [0.0 0.45 0.74];   % blue
COL_FREQ = [0.20 0.20 0.20]; % dark gray
LINE_W   = 1.8;

if ~exist(OUT_ROOT,"dir"), mkdir(OUT_ROOT); end

for c = 1:numel(cutters)
    cutter = cutters(c);
    fprintf("\nProcessing %s ...\n", cutter);

    cutter_dir = fullfile(DATA_ROOT, cutter);
    if ~exist(cutter_dir,"dir"), continue; end

    % -----------------------------------------------------------
    % Find signal files recursively (exclude wear)
    % -----------------------------------------------------------
    files = [ ...
        dir(fullfile(cutter_dir,"**","*.csv")); ...
        dir(fullfile(cutter_dir,"**","*.txt")); ...
        dir(fullfile(cutter_dir,"**","*.mat")) ];

    files = files(~contains(lower({files.name}),"wear"));
    if isempty(files), continue; end

    % Sort numerically
    nums = nan(numel(files),1);
    for i = 1:numel(files)
        tok = regexp(files(i).name,'\d+','match','once');
        if ~isempty(tok), nums(i) = str2double(tok); end
    end
    nums(isnan(nums)) = inf;
    [~,idx] = sort(nums);
    files = files(idx);

    % Mid-life cut
    k = round(numel(files)/2);
    data_file = fullfile(files(k).folder, files(k).name);
    fprintf("Using file: %s\n", files(k).name);

    % -----------------------------------------------------------
    % Load signal
    % -----------------------------------------------------------
    [~,~,ext] = fileparts(data_file);
    ext = lower(ext);

    if ext == ".csv" || ext == ".txt"
        raw = readmatrix(data_file);
        signal = raw(:,sensor_idx);
    else
        S = load(data_file);
        fn = fieldnames(S);
        signal = [];
        for j = 1:numel(fn)
            v = S.(fn{j});
            if isnumeric(v)
                if isvector(v)
                    signal = v(:); break;
                elseif size(v,2) >= sensor_idx
                    signal = v(:,sensor_idx); break;
                end
            end
        end
        if isempty(signal), error("Cannot extract signal"); end
    end

    signal = signal(1:min(end,N));
    t = (0:numel(signal)-1)/FS;

    OUT_DIR = fullfile(OUT_ROOT,cutter);
    if ~exist(OUT_DIR,"dir"), mkdir(OUT_DIR); end

    %% ================= TIME DOMAIN =================
    fig = figure("Color","w","Position",[200 200 900 260]);
    plot(t, signal, "Color",COL_TIME, "LineWidth",LINE_W);
    grid on;
    ax = gca;
    ax.FontWeight = "bold";
    ax.XAxis.FontWeight = "bold";
    ax.YAxis.FontWeight = "bold";
    exportgraphics(fig, fullfile(OUT_DIR,"time_domain.png"), "Resolution",600);
    close(fig);

    %% ================= FREQUENCY DOMAIN =================
    Y = abs(fft(signal));
    f = (0:length(Y)-1)*FS/length(Y);

    fig = figure("Color","w","Position",[200 200 900 260]);
    plot(f(1:floor(end/2)), Y(1:floor(end/2)), ...
         "Color",COL_FREQ, "LineWidth",LINE_W);
    grid on;
    ax = gca;
    ax.FontWeight = "bold";
    ax.XAxis.FontWeight = "bold";
    ax.YAxis.FontWeight = "bold";
    exportgraphics(fig, fullfile(OUT_DIR,"frequency_domain.png"), "Resolution",600);
    close(fig);

    %% ================= CWT =================
    figure("Color","w","Position",[200 200 900 380]);
    cwt(signal,FS);
    ax = gca;
    ax.FontWeight = "bold";
    ax.XAxis.FontWeight = "bold";
    ax.YAxis.FontWeight = "bold";
    fig = gcf;  % 🔴 IMPORTANT FIX
    exportgraphics(fig, fullfile(OUT_DIR,"cwt.png"), "Resolution",600);
    close(fig);

    fprintf("✅ Saved plots for %s\n", cutter);
end

fprintf("\n✅ ALL CUTTERS c1–c6 DONE SUCCESSFULLY\n");
