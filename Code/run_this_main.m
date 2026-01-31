% ===================================================================
% WFS DATA READER TOOLKIT - FOR PCI AE FILES (.wfs)
% Author: UMAR's Assistant
% Purpose: Read, decode, explore and extract info from AE .wfs files
% Dataset: Concrete RUL (Acoustic Emission, 81 GB, 8-channel, 5 MHz)
% ===================================================================

%% MASTER FILE - run_this_main.m
% This is the ONLY file users run. Everything else is called from here.

clc; clear;

% === USER SETTING ===
filePath = "D:\\Concrete RUL\\STREAM20191203-105108-756.wfs";
start_time = 0;      % seconds
duration = 1;        % read 1 second
channels = 1:8;      % all AE channels

% === CALL MAIN READER ===
[signals, t, fs, nch, header_info] = wfsread_exp_fullinfo(filePath, start_time, duration, channels);

% === BASIC OUTPUT ===
fprintf('\n===== WFS File Information =====\n');
fprintf('File        : %s\n', filePath);
fprintf('Channels    : %d\n', nch);
fprintf('Sampling Hz : %.0f\n', fs);
fprintf('Max Voltage : %.2f V\n', header_info.Max_voltage);
fprintf('Header Size : %d bytes\n', header_info.Header_length);
fprintf('Pretrigger  : %d samples\n', header_info.Pretrigger);
fprintf('Signal Size : %d rows × %d columns\n', size(signals,1), size(signals,2));

% === STATS PER CHANNEL ===
fprintf('\nBasic Stats Per Channel (First %d sec):\n', duration);
for ch = 1:nch
    ch_data = signals(:,ch);
    fprintf('Channel %d → RMS: %.4f | Peak: %.4f | Min: %.4f | Max: %.4f\n', ch, rms(ch_data), max(abs(ch_data)), min(ch_data), max(ch_data));
end

% === ESTIMATED DURATION ===
f = dir(filePath);
file_bytes = f.bytes;
est_duration_sec = floor((file_bytes - header_info.Header_length) / (fs * nch * 2));
fprintf('\nEstimated Total Duration: ~%d seconds (%.2f hours)\n', est_duration_sec, est_duration_sec/3600);


% ===================================================================
% SUPPORT FILE: wfsread_exp_fullinfo.m
% Reusable function to extract data and metadata from .wfs file
% ===================================================================
function [signals,t,fs,nch,header] = wfsread_exp_fullinfo(filename, start_time, end_time, channels)

    if nargin < 4
        channels = 1:8;
    end

    % -- Read header --
    [nch, fs_kHz, max_v, hdr_len, delay_idx, pretrig] = PCI2ReadHeader(filename);
    fs = fs_kHz * 1000;

    header = struct("Max_voltage", max_v, "Header_length", hdr_len, "Pretrigger", pretrig);

    % -- Read signals for each channel --
    samples_to_read = round((end_time - start_time) * fs);
    signals = zeros(samples_to_read, length(channels));

    for i = 1:length(channels)
        ch = channels(i);
        fid = fopen(filename, 'rb');
        if fid < 0, error("Cannot open file"); end
        offset = hdr_len + (ch - 1) * 8220 + 28 + 2 * ch;
        fseek(fid, offset + round(fs * start_time * 2), 'bof');
        raw = fread(fid, samples_to_read, 'short', (8222 * (nch - 1)));
        fclose(fid);
        signals(:, i) = (max_v / 32767) * double(raw);
    end

    t = (start_time : 1/fs : end_time - 1/fs)';
end


% ===================================================================
% SUPPORT FILE: PCI2ReadHeader.m
% Parses PCI header metadata
% ===================================================================
function [nch, fs_kHz, max_v, hdr_len, delay_idx, pretrig] = PCI2ReadHeader(filename)
    fid = fopen(filename,'rb');
    if fid < 0, error('Cannot open file'); end

    fread(fid,1,'short'); fseek(fid, 3, 'cof');
    nch = fread(fid,1,'int8');
    fseek(fid, -4, 'cof');
    fread(fid,1,'short');
    fseek(fid, 12, 'cof');
    fs_kHz = fread(fid,1,'short');
    fread(fid,1,'short');
    fread(fid,1,'short');
    pretrig = fread(fid,1,'short');
    fseek(fid, 2, 'cof');
    max_v = fread(fid,1,'short');
    hdr_len = 296; % fixed size based on inspection
    delay_idx = 0;
    fclose(fid);
end
