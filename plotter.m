%Laden der Daten. Pfad austauschen.%
load("C:\Users\Raoul\112-Medizin\Training\training\train_ecg_00001.mat");
%Erzeugen der Zeit Menge t.%
x = size(val);
t = (1:x(2));
%Plotten des elektrischen Potentials über die Zeit.%
figure;
plot(t, val(1,:))
xlabel('Zeit in ?')
ylabel('Spannung in mV')
title('Daten aus EKG')
hold on;
load("C:\Users\Raoul\112-Medizin\Training\training\train_ecg_00010.mat");
x = size(val);
t = (1:x(2));
plot(t, val(1,:))
legend('1', 'n')
%Wenn zwei verschiedene EKG's in einem Plot verglichen werden sollen hier
%zusätzlich Pfad ändern und die Zeilen auskommentieren.%


%load('C:\Users\Raoul\112---Medizin\training\train_ecg_00002.mat');
%Erzeugen der Zeit Menge t%
%x = size(val);
%t = (1:x(2));
%plot(t, val(1,:))