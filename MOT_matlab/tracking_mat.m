%function tracking_mat(bboxes_tracked)
fid = fopen('tracking_CNN.txt','a+');

for fr=1:length(bboxes_tracked)
    t = bboxes_tracked(fr).bbox;

    for i=1:size(t,1)
        if (length(t)~=0)
            x = t(i,1);
            y = t(i,2);
            w = t(i,3)-t(i,1);
            h = t(i,4)-t(i,2);
            fprintf(fid,'%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1\n',fr,t(i,5),x,y,w,h);      
        end 
    end
end
fclose(fid);
% 
