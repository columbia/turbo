CREATE DATABASE covid;

\c covid

CREATE TABLE covid_data (
    time        INT               NOT NULL,
    positive    INT               NOT NULL,
    gender      INT               NOT NULL,
    age         INT               NOT NULL,
    ethnicity   INT               NOT NULL
);

SELECT create_hypertable('covid_data', 'time', chunk_time_interval => 1);

CREATE DATABASE citibike;

\c citibike

CREATE TABLE citibike_data (
    time              INT               NOT NULL,
    weekday           INT               NOT NULL,
    hour              INT               NOT NULL,
    duration_minutes  INT               NOT NULL,
    start_station     INT               NOT NULL,
    end_station       INT               NOT NULL,
    usertype          INT               NOT NULL,
    gender            INT               NOT NULL,
    age               INT               NOT NULL
);

SELECT create_hypertable('citibike_data', 'time', chunk_time_interval => 1);