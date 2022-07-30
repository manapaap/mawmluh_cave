% Twsting frequency analysis on here
% Let's load some data and see what happens

folder = 'C:/Users/Aakas/Documents/School/Oster_lab/';
file = 'internal_excel_sheets/filled_seb_runs/MAW-3-downsample.csv';

maw_3_proxy = readtable([folder ,file]);

num_years = 20;

% Plot adjustment stuff
[maxf,minf] = cwtfreqbounds(numel(maw_3_proxy.d18O),years(num_years));
numfreq = 10;
freq = logspace(log10(years(minf)),log10(years(maxf)),numfreq);
freq = cast(freq, 'int32');

cwt(maw_3_proxy.d18O, years(num_years))
% cwt(maw_3_proxy.d13C, years(num_years))

AX = gca;
AX.YTickLabelMode = "auto";
AX.YTick = freq;

