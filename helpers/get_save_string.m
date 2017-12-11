function [save_string] = get_save_string(save_string_try)

if(exist(save_string_try,'file') == 0 && exist(strcat(save_string_try,'.mat'),'file') == 0)
    save_string = save_string_try;
else
    n = 1;
    save_string = sprintf('%s_%d',save_string_try,n);
    while(exist(save_string,'file') >0 || exist(strcat(save_string,'.mat'),'file') > 0)
        n = n + 1;
        save_string = sprintf('%s_%d',save_string_try,n);
    end
end
end

