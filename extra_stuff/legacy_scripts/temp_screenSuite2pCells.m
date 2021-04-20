clear dFoF
for ii = 1:size(F,1)
    dFoF(ii,:) = F(ii,:) ./ prctile(F(ii,:),20);
end

for ii = fliplr(1:size(dFoF,1))
    if iscell(ii,1) == 1
        if  sum(spks(ii,:) > 10) > 50
            if  sum(spks(ii,:) > 100) > 10
                %                 plot(zscore(dFoF(ii,:))/3+ii,'k')
                % plot(spks(ii,:)+ii*50)
                tmpMinY = min(stat{ii}.ypix);
                tmpMaxY = max(stat{ii}.ypix);
                tmpMinX = min(stat{ii}.xpix);
                tmpMaxX = max(stat{ii}.xpix);
                
                tmpIm = double(ops.meanImg);
                tmpIm2 = double(ops.sdmov);
                tmpIm3 = double(ops.meanImg);
                tmpIm4 = double(ops.meanImgE);
                tmpIm5 = double(ops.max_proj);
                tmpIm6 = double(ops.Vcorr);
                
                for jj = 1:stat{ii}.npix
                    tmpIm(stat{ii}.ypix(jj)+1 , stat{ii}.xpix(jj)+1) = stat{ii}.lam(jj)*100;
                end
                %                 tmpIm(stat{ii}.ypix , stat{ii}.xpix) = 1000;
                fig = figure;
%                 fig.Position = [100 50 150 900];
                fig.Position = [10 50 1200 800];
                fig.Name = ['cell #: ' , num2str(ii)];
                
                subplot(3,6,1)
                imagesc(tmpIm(tmpMinY:tmpMaxY+1 , tmpMinX:tmpMaxX+1))
                title(['mask cell# ' , num2str(ii)])
                
                subplot(3,6,2)
                imagesc(tmpIm2(tmpMinY:tmpMaxY+1 , tmpMinX:tmpMaxX+1))
                title(num2str(stat{ii}.aspect_ratio))
                title('sdmov')
                
                subplot(3,6,3)
                imagesc(tmpIm3(tmpMinY:tmpMaxY+1 , tmpMinX:tmpMaxX+1))
                title('meanImg')
                
                subplot(3,6,4)
                imagesc(tmpIm4(tmpMinY:tmpMaxY+1 , tmpMinX:tmpMaxX+1))
                title('meanImgE')
                
                subplot(3,6,5)
                imagesc(tmpIm5(tmpMinY:tmpMaxY+1 , tmpMinX:tmpMaxX+1))
                title('max_proj')
                
                subplot(3,6,6)
                imagesc(tmpIm6(tmpMinY:tmpMaxY+1 , tmpMinX:tmpMaxX+1))
                title('Vcorr')
                
%                 fig2 = figure;
                subplot(3,6,7:18)
                plot(dFoF(ii,:))
%                 fig2.Name = num2str(ii);
                
                %                 figure; imagesc(ops.meanImg(stat{ii}.ypix , stat{ii}.xpix))
            end
        end
    end
end

fig2=figure;
fig2.Name = ['mean F of all cells'];
plot(mean(F,1))
