Fs = 2.32;
%%
figure; hold on
for ii = 1:size(F,1)
    plot((1:size(F,2)) / Fs, ii*5+ zscore(F(ii,:)))
end


%%
clear dFoF
for ii=1:size(F,1)
%     dFoF(ii,:) = (F(ii,:) - prctile(F(ii,:),30)) ./ prctile(F(ii,:),30);
    dFoF(ii,:) = (F(ii,:)' - smooth(F(ii,:),100)) ./ prctile(F(ii,:),30);
    dFoF_zscore(ii,:) = zscore(dFoF(ii,:));
end

figure; imagesc(dFoF_zscore)
%%
figure; hold on
for ii = 1:size(dFoF,1)
    plot((1:size(dFoF,2)) / Fs, ii*2+ (dFoF(ii,:)),'k')
end

%%
hey = pca(dFoF_zscore);

figure; 
for ii = 1:5
    subplot(5,1,ii)
plot(hey(:,ii))
end