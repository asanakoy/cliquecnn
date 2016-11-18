function [] = getImagePaths( category )
    %GETIMAGEPATHS Get image pathes for the OlympicSports category
    %  Write pathes to file

    mat = load(['../../similarities_hog_lda/simMatrix_',category,'.mat'], 'image_names');
    file_name = ['./imagePaths_',category,'.txt'];

    imnames = zeros(size(mat.image_names));
    for i = 1:size(mat.image_names,1)
        fprintf('%d / %d\n', i, size(mat.image_names, 1));
        [path, fname, ext] = fileparts(mat.image_names(i,:));
        frame_name = str2double(fname(2:end))-1;
        n_name = [path '/' sprintf('I%05d',frame_name) ext];
        imnames(i,:) = n_name;
    end

    fileID = fopen(file_name,'w');
    for i = 1:size(imnames,1)
        fprintf(fileID,'%s\n',imnames(i,:));
    end
    fclose(fileID);
end

