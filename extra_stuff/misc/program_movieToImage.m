%% Make image for reference

%% Manual Import
directory = 'F:\RH_Local\Rich data\registration images\mouse 10.30 image for registration';
movie_all = bigread4([directory, '\baseline_00001_00001.tif']);

%%
Fs_frameRate = 30; % in Hz
duration_trace = size(movie_all,3) / Fs_frameRate;
duration_trial = 30; % in seconds
baseline_pctile = 20;
%% Make Standard Deviation Image
tic
chunk_size = 25; % number of columns to process at once. vary this value to maximize speed. There is a sweet spot for memory usage around 25 columns of size 512 each.
movie_std = nan(size(movie_all,1), size(movie_all,2));
for ii = 1:chunk_size:size(movie_all,2)
    if ii + chunk_size > size(movie_all,2)
        movie_std(:,ii:size(movie_all,2)) = std(single(movie_all(:,ii:size(movie_all,2),:)),[],3);
    else
        movie_std(:,ii:ii+chunk_size) = std(single(movie_all(:,ii:ii+chunk_size,:)),[],3);
    end
end
toc
%%
movie_mean = mean(movie_all,3);
movie_fano = movie_std ./ movie_mean;
movie_clahe = adapthisteq(movie_mean./max(movie_mean(:)),'NumTiles',[20 20],'clipLimit',0.2,'Distribution','rayleigh');

h1 = figure;
imagesc(movie_mean)

h2 = figure;
imagesc(movie_fano)
set(gca,'CLim',[0 1])