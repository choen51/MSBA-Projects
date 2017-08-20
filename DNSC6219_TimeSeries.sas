
/***************************************************************************************
 *   PRODUCT:   SAS                                                                    *
 *   VERSION:   9.4                                                                    *
 *   CREATOR:   Kelly Berdelle, Daniel Chen, Kevin Fizgerald, Aida Rojas, Xing Zhang   *
 *   DATE:      May 15,2017                                                            *
 *   DESC:      Time Series Final Project Part 2:  Multivariate Time Series Analysis   *
 ***************************************************************************************/

DATA Energy;
	Set SASUSER.SEDS(obs=180);  /*HOLDING OUT 12 OBSERVATIONS*/
	reformatted_month=month(Month);
TIME=_N_;

/*We identified NATURAL GAS as our output series and COAL and HYDROELECTRIC as our input series.*/

/*OVERVIEW: ACFs, PACFs, IACFs, correlations of variables, and autocorrelation check of residuals*/

PROC ARIMA DATA=Energy;
	IDENTIFY VAR=natural_gas;
RUN;

PROC ARIMA DATA=Energy;
	IDENTIFY VAR=coal;
RUN;

PROC ARIMA DATA=Energy;
	IDENTIFY VAR=hydroelectric;
RUN;

PROC ARIMA DATA=Energy;
	I VAR=natural_gas CROSSCOR=(coal hydroelectric) NOPRINT;
	E INPUT=(coal hydroelectric) PLOT;
RUN;

/*OVERVIEW:  Boxplots*/
 
PROC SORT DATA=Energy;
	BY reformatted_month;

PROC BOXPLOT;
	PLOT natural_gas*reformatted_month;
	TITLE ’Seasonal Box Plot for Natural Gas Power Output in GWh’;
RUN;

PROC BOXPLOT;
	PLOT coal*reformatted_month;
	TITLE ’Seasonal Box Plot for Coal Power Output in GWh’;
RUN;

PROC BOXPLOT;
	PLOT hydroelectric*reformatted_month;
	TITLE ’Seasonal Box Plot for Hydroelectric Power Output in GWh’;
RUN;

/*STEP 1: CHECK FOR STATIONARITY: output series NATURAL GAS*/
PROC ARIMA DATA=Energy;
	IDENTIFY VAR=natural_gas;
RUN;

/*STEP 1: CHECK FOR STATIONARITY: input series COAL*/
PROC ARIMA DATA=Energy;
	IDENTIFY VAR=coal;
RUN;

/*STEP 1: CHECK FOR STATIONARITY: input series HYDROELECTRIC*/
PROC ARIMA DATA=Energy;
	IDENTIFY VAR=hydroelectric;
RUN;

/*Series are not stationary. Induce stationarity through differencing.*/

/*STEP 2: DIFFERENCING*/
PROC ARIMA DATA=Energy;
	IDENTIFY VAR=natural_gas(1);
	IDENTIFY VAR=coal(1);
	IDENTIFY VAR=hydroelectric(1);
RUN;

/*Differenced COAL, NATURAL GAS, and HYDROELECTRIC are not stationary and not white noise. 
Perform pre-whitening.*/

/*STEP 3: PREWHITENING: NATURAL GAS*/
DATA Energy;
	Set SASUSER.SEDS(obs=180);  /*HOLDING OUT 12 OBSERVATIONS*/
	reformatted_month=month(Month);
TIME=_N_;

PROC ARIMA DATA=Energy;
	IDENTIFY VAR=natural_gas(12) NOPRINT;
	ESTIMATE P=(1) Q=(12) METHOD=ML;  /*ARIMA(1,0,0)(0,1,1)^12 + C*/  
	FORECAST LEAD=12 OUT=results;
RUN;  /*White noise! Maximum Likelihood produces better result than Unconditional Least Squares*/

/*STEP 3: PREWHITENING: COAL*/

PROC ARIMA DATA=Energy;
	IDENTIFY VAR=coal(12) NOPRINT;
	ESTIMATE P=(1) Q=(12) METHOD=ML;  /*ARIMA(1,0,0)(0,1,1)^12 + C*/  
RUN;  /*White noise! Maximum Likelihood produces better result than Unconditional Least Squares*/

/*STEP 3: PREWHITENING: HYDROELECTRIC*/

PROC ARIMA DATA=Energy;
	IDENTIFY VAR=hydroelectric(12) NOPRINT;
	ESTIMATE P=(1)(12) NOINT METHOD=ML;  /*ARIMA(1,0,0)(1,1,0)^12 NOINT*/ 
RUN;  /*White noise!*/

/*STEP 4: 1-INPUT TRANSFER FUNCTION MODEL 1: NATURAL GAS vs COAL*/

PROC ARIMA DATA=ENERGY;
	IDENTIFY VAR=natural_gas(12) NOPRINT;
	IDENTIFY VAR=coal(12) NOPRINT;
	ESTIMATE P=(1) Q=(12) METHOD=ML;
	IDENTIFY VAR=natural_gas(12) CROSSCOR=(coal(12));
RUN;
/*The ACF is stationary except at lag 12.*/
/*The CCF shows first response at lag 0, drop at lag 1, and no patterns: b=0, r=0, s=1*/

PROC ARIMA DATA=ENERGY;
	IDENTIFY VAR=natural_gas(12) NOPRINT;
	IDENTIFY VAR=coal(12) NOPRINT;
	ESTIMATE P=(1) Q=(12) METHOD=ML;
	IDENTIFY VAR=natural_gas(12) CROSSCOR=(coal(12)) NOPRINT;
	ESTIMATE INPUT=((1)/coal) P=(1) Q=(12) METHOD=ML PLOT;  /*b=0, r=0, s=1*/
	FORECAST LEAD=12 OUT=results;
RUN;

/*STEP 5: ADEQUACY CHECK:  NATURAL GAS vs COAL*/
/*The error model behaves white noise.*/

/*STEP 6: 1-INPUT TRANSFER FUNCTION MODEL 2: NATURAL GAS vs HYDROELECRTRIC*/

PROC ARIMA DATA=ENERGY;
	IDENTIFY VAR=natural_gas(12) NOPRINT;
	IDENTIFY VAR=hydroelectric(12) NOPRINT;
	ESTIMATE P=(1)(12) NOINT METHOD=ML;  /*ARIMA(1,0,0)(1,1,0)^12 NOINT*/
	IDENTIFY VAR=natural_gas(12) CROSSCOR=(hydroelectric(12));
RUN;
/*The ACF is stationary except at lag 12.*/
/*The CCF shows no responses at any lags, no exponential decay, and no patterns: d=0, r=0, s=0*/

/*STEP 7: 2-INPUT TRANSFER FUNCTION MODEL: NATURAL GAS vs COAL vs HYDROELECRTRIC*/

/*PROC ARIMA DATA=ENERGY;
	IDENTIFY VAR=natural_gas(12) NOPRINT;
	IDENTIFY VAR=coal(12) NOPRINT;
	ESTIMATE P=(1) Q=(12) METHOD=ML NOPRINT;
	IDENTIFY VAR=hydroelectric(12) NOPRINT;
	ESTIMATE P=(1)(12) NOINT METHOD=ML NOPRINT;
	IDENTIFY VAR=natural_gas(12) CROSSCOR=(coal(12) hydroelectric(12)) NOPRINT;
	ESTIMATE INPUT=((1)/coal, 1$/hydroelectric) P=(1) Q=(12) METHOD=ML PLOT;  
RUN;*/


