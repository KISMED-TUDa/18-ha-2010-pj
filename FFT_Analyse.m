%FFT-Analyse%
load("C:\Users\Raoul\112-Medizin\Training\training\train_ecg_00001.mat");
x = size(val);
t = (1:x(2));
%figure;
%plot(t, val(1,:))
Y = fft(val);
P2 = abs(Y/x(2));
P1 = P2(1:(x(2))/2+1);
P1(2:end-1) = 2*P1(2:end-1);
Fs = 10;
f = Fs*(0:(x(2)/2))/x(2);
figure;
plot(f,P1)
%----%
load("C:\Users\Raoul\112-Medizin\Training\training\train_ecg_00010.mat");
x = size(val);
t = (1:x(2));
hold on;
%figure;
%plot(t, val(1,:))
Y = fft(val);
P2 = abs(Y/x(2));
P1 = P2(1:(x(2))/2+1);
P1(2:end-1) = 2*P1(2:end-1);
Fs = 10;
f = 1000*(0:(x(2)/2))/x(2);
figure;
plot(f,P1)