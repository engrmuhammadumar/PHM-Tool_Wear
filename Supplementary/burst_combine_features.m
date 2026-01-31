input_dir = 'F:\concrete test 1\';
output_file = 'F:\concrete test 1\AE_burst_features_all_chunks.csv';

threshold_factor = 5;  % adjustable burst detection threshold
fs = 5e6;              % 5 MHz sampling rate (already known)

% Create or clear the output file
fid_all = fopen(output_file, 'w');
fprintf(fid_all, 'chunk_id,start_idx,end_idx,duration,rise_time,peak_amp,energy\n');

for i = 0:1036
    try
        % Load 1-second chunk
        file_name = sprintf('chunk_%04d.mat', i);
        load(fullfile(input_dir, file_name), 'signals');
        
        ch = signals(:, 1);  % Channel 1 only
        rms_val = rms(ch);
        threshold = threshold_factor * rms_val;
        
        % Detect burst samples
        burst_idx = find(abs(ch) > threshold);
        if isempty(burst_idx)
            continue;
        end
        
        % Segment bursts based on gap
        burst_diff = diff(burst_idx);
        burst_split = [0; find(burst_diff > fs * 0.001); length(burst_idx)];  % gap > 1 ms
       
        % Extract features for each burst
        for j = 1:length(burst_split) - 1
            burst_range = burst_idx(burst_split(j)+1 : burst_split(j+1));
            if length(burst_range) < 10
                continue;
            end
            burst_signal = ch(burst_range);
            start_idx = burst_range(1);
            end_idx = burst_range(end);
            peak_amp = max(abs(burst_signal));
            energy = sum(burst_signal.^2);
            duration = (end_idx - start_idx) / fs;
            rise_time = (find(abs(burst_signal) == peak_amp, 1) - 1) / fs;

            % Write to combined file
            fprintf(fid_all, '%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n', ...
                i, start_idx, end_idx, duration, rise_time, peak_amp, energy);
        end
        
        fprintf('✅ Processed chunk %d\n', i);

    catch ME
        warning('❌ Error on chunk %d: %s', i, ME.message);
    end
end

fclose(fid_all);
disp('🎉 All burst features extracted and combined into one file!');
