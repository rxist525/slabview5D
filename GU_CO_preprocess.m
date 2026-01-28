%% load database
clear

if ispc
    rt = 'V:\Gokul\ImageAnalysis\gitRepos\slabview5D';
    addpath(genpath('V:\Gokul\ImageAnalysis\gitRepos\XR_Repository'));
    addpath(genpath('V:\Gokul\ImageAnalysis\gitRepos\XR_GU_Repository'));
    addpath(genpath('V:\Gokul\ImageAnalysis\gitRepos\PetaKit5D')); %% need this for the denoising
    jobLogDir = 'V:\Gokul\joblogs';
else
    rt = '/clusterfs/nvme/Gokul/ImageAnalysis/gitRepos/slabview5D';
    addpath(genpath('/clusterfs/nvme/ABCcode/XR_Repository'));
    addpath(genpath('/clusterfs/nvme/ABCcode/GU_XR_Repository'));
    addpath(genpath('/clusterfs/nvme/ABCcode/PetaKit5D')); %% need this for the denoising
    jobLogDir = '/clusterfs/nvme/Gokul/joblogs';
end

fn = 'CO_database_20260127_145210.mat';
load([rt filesep fn]);

%% skip chromatic offset correction
%% skip unmixing
%% skip deconvolution

%% 
if ispc
    dataPath = {CO_db.winpath}';
else
      dataPath = {CO_db.path}';
end
Overwrite = true;

for k = 1:numel(CO_db)
    sfn = numel(dir('3D settings*.csv')) > 0;
    if sfn 
        if Overwrite || ~exist('ImageList_from_sqlite.csv', 'file')
        tic
        imageListFullpath = stitch_generate_imagelist_from_sqlite(dataPath{k});
        toc
        CO_db(k).stitch_generate_imagelist_from_sqlite = true;
        CO_db(k).skipStitching = false;
        CO_db(k).stitchingImageListPath = [dataPath{k} filesep 'ImageList_from_sqlite.csv'];
        end
    elseif CO_db(k).processedStitchingImageList

    else
        CO_db(k).stitch_generate_imagelist_from_sqlite = false;
        CO_db(k).skipStitching = true;
    end
    CO_db(k).processedStitchingImageList = true;
    k
end