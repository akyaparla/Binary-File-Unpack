function [data, t] = loaddata(fileName, numChan, fs)
    fd = fopen(fileName, "rb");
    data = fread(fd, [numChan, inf], "float32");
    fclose(fd);
    dt = 1/fs;
    t = 0:dt:(length(data)-1)*dt;
end

    
