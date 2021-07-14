function bmass_matlab_main()
fp = fopen('params.txt', 'r');
input_path = fscanf(fp,'%s',1);
output_path = fscanf(fp,'%s',1);
odds = fscanf(fp,'%f',1);

[files, file_num] = files_finding(input_path);


for i = 1:file_num
    file_path = strcat(input_path, files{i});
    data = xlsread(file_path);
    time = (1:size(data,1))'/10;
    
    seg = bmass(time, data, odds);
    
    seg = seg_boundaries(seg);
    file_save_path = strcat(output_path, files{i});
    xlswrite(file_save_path, seg);
end
end     
        
%% Files finding
function [files,len] = files_finding(path)
files_dir = dir(path);
len = length(files_dir)-2;
files = cell(len,1);
for i = 1:len
    files{i} = files_dir(i+2).name;
end
end


function segments = seg_boundaries(seg)
[m,~] = size(seg);
segments = zeros(m+1,1);
for i = 2:m+1
    segments(i) = segments(i-1)+ length(seg{i-1});
end
segments(end) = segments(end)-1;
end