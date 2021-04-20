function hash = compute_hashSequence(image_sequence, waitbar_pref)

if ~exist('waitbar_pref')
    waitbar_pref=0;
end
if waitbar_pref
    wbar = waitbar(0, 'creating hash sequence');
end

n_im = size(image_sequence,3); % number of images in sequence
hash = nan(n_im,1);

tic
for ii = 1:n_im
    hash(ii) = simple_image_hash(image_sequence(:,:,ii));
    if mod(ii,100)==0 && waitbar_pref
        waitbar(ii/n_im, wbar, 'creating hash sequence');
    end      
end
duration = toc;
if waitbar_pref
    disp([num2str(round((duration/n_im)*1000,4)) , ' ms/iteration'])
    close(wbar)
end

end