function [] = BBplotSettings(fs,grid_yn)
set(gca,'FontSize',fs,'FontName','times','YColor','k')
if grid_yn
    grid on
end
end