clear all;
filepath = "C:\Users\EN301\Desktop\Rong\Phone_Lock_RX65N\Sample\Screen_Test";
keyword = "";
fileFolder=fullfile(filepath);%資料夾名plane
dirOutput=dir(fullfile(fileFolder,keyword+'*.csv'));%如果存在不同型別的檔案，用‘*’讀取所有，如果讀取特定型別檔案，'.'加上檔案型別，例如用‘.jpg’
fileNames={dirOutput.name}';

Data = zeros(750,8192);
for i = 1:750
    A = csvread(filepath+"\"+fileNames(i),0,0);
    A(1) = 0;
    Data(i,:) = A;
end

csvwrite(filepath+"\Test.csv",Data,1,0);