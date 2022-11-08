with icu_order as (
    select
        *,
        ROW_NUMBER() OVER (partition by HADM_ID
            order by OUTTIME desc
        ) as icu_order_out,
        ROW_NUMBER() OVER (partition by HADM_ID
            order by INTIME
        ) as icu_order_in
    from
        `physionet-data.mimiciii_clinical.icustays`
), adm_by_tsf as (
    select
        HADM_ID, INTIME
    from
        `physionet-data.mimiciii_clinical.transfers`
    where
        EVENTTYPE = 'admit'
)
, disch_by_tsf as (
    select
        HADM_ID, INTIME
    from
        `physionet-data.mimiciii_clinical.transfers`
    where
        EVENTTYPE = 'discharge'
), adm_by_icu as (
    select
        HADM_ID, INTIME, OUTTIME
    from
        icu_order
    where
        icu_order_in = 1
), disch_by_icu as (
    select 
        HADM_ID, OUTTIME
    from
        icu_order
    where
        icu_order_out = 1
), admittime as (
    select
        a.SUBJECT_ID, a.HADM_ID,
        case 
            when i.INTIME< a.ADMITTIME then i.INTIME 
            when i.INTIME>=a.ADMITTIME and t.INTIME < a.ADMITTIME then t.INTIME 
            else a.ADMITTIME
        end as ADMIT_TIME
    from 
        `physionet-data.mimiciii_clinical.admissions` a
        left join adm_by_icu i on a.HADM_ID = i.HADM_ID
        left join adm_by_tsf t on a.HADM_ID = t.HADM_ID
), dischtime as (
    select 
        a.HADM_ID,
        case 
            when i.OUTTIME > a.DISCHTIME then i.OUTTIME
            when i.OUTTIME <=a.DISCHTIME and t.INTIME > a.DISCHTIME then t.INTIME 
            else a.DISCHTIME
        end as DISCHARGE_TIME
    from 
        `physionet-data.mimiciii_clinical.admissions` a
        left join disch_by_icu i on a.HADM_ID = i.HADM_ID
        left join disch_by_tsf t on a.HADM_ID = t.HADM_ID
), adm_disc_time as (
    select
        t1.SUBJECT_ID, t1.HADM_ID, ADMIT_TIME, DISCHARGE_TIME, t3.INTIME ICU_INTIME, t3.OUTTIME ICU_OUTTIME
    from 
        admittime t1 
        inner join dischtime t2 on t1.HADM_ID = t2.HADM_ID
        left join adm_by_icu t3 on t1.HADM_ID = t3.HADM_ID
), los as (
    select 
        HADM_ID, 
        DATETIME_DIFF(DISCHARGE_TIME, ICU_INTIME, MINUTE) / 60 LOS_HOUR_REST,
        DATETIME_DIFF(DISCHARGE_TIME, ADMIT_TIME, MINUTE) / 60 LOS_HOUR_HOSP,
        DATETIME_DIFF(ICU_OUTTIME, ICU_INTIME, MINUTE) / 60 LOS_HOUR_ICU
    from 
        adm_disc_time
), age as (
    select 
        t.HADM_ID, DATETIME_DIFF(t.ADMIT_TIME, p.DOB, DAY) / 365 age
    from
        adm_disc_time t 
        left join `physionet-data.mimiciii_clinical.patients` p on t.SUBJECT_ID = p.SUBJECT_ID
), icu_ct as (
    select
        HADM_ID, count(ICUSTAY_ID) count_icu
    from
        `physionet-data.mimiciii_clinical.icustays`
    group by 
        HADM_ID
), icu_view as (
    select 
        SUBJECT_ID, i.HADM_ID, ICUSTAY_ID, INTIME, OUTTIME
    from 
        icu_order i
        left join icu_ct c on i.HADM_ID = c.HADM_ID
    where 
        icu_order_in = 1 and count_icu = 1
), note_time as (
    select
        t.HADM_ID,
        DATETIME_DIFF(n.CHARTTIME, t.ICU_INTIME, MINUTE) / 60 ctime_icu,
        DATETIME_DIFF(n.CHARTDATE, t.ICU_INTIME, MINUTE) / 60 cdate_icu
    from
        adm_disc_time t
        left join `physionet-data.mimiciii_notes.noteevents` n on t.HADM_ID = n.HADM_ID
    where
        n.CATEGORY != 'Discharge summary'
), note_crit as (
    select 
        distinct HADM_ID
    from 
        note_time
    where 
        ctime_icu <= 48 or cdate_icu <= 48
)
select 
    d.SUBJECT_ID, d.HADM_ID, i.ICUSTAY_ID, i.INTIME, i.OUTTIME, d.DRG_CODE
from 
    `physionet-data.mimiciii_clinical.drgcodes` d
    inner join note_crit n on n.HADM_ID = d.HADM_ID
    inner join icu_view i on i.HADM_ID = d.HADM_ID
    left join los l on l.HADM_ID = d.HADM_ID
    left join age a on a.HADM_ID = d.HADM_ID
where 
    DRG_TYPE = 'APR ' and d.DESCRIPTION is not null
    and a.age > 15 and l.LOS_HOUR_REST > (48 + 24) # pred window + 24hr
order by 
    d.SUBJECT_ID, d.HADM_ID, i.ICUSTAY_ID


# 45304 queried on 09/09/2022; need to remove duplicates etc
