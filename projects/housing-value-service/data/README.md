# Data snapshot

The unchanged `housing.csv` snapshot contains 20,640 historical California census
district records, 8 raw numerical features, one categorical feature and the
`median_house_value` target in US dollars. There are 207 missing bedroom counts
and no fully duplicated rows in this snapshot.

Distribution source: Aurélien Géron's Hands-On Machine Learning repository:
https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv

Source Git blob: `d5e3fa249979387797b225dee7dc2c18e619c42c`.

Snapshot SHA-256:
`8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e`.

The distribution repository's Apache 2.0 license is retained in
`SOURCE-LICENSE.txt`. Source attribution is retained here and in the demo.
The original dataset draws on the 1990 California census and the California
housing regression benchmark associated with Pace and Barry.

Median income is recorded in scaled dataset income units, approximately tens
of thousands of US dollars. Household, population, room and bedroom fields are
district totals. They are not individual-house measurements.

The target is capped near USD 500,001. Housing age is capped at 52 years.
These limits affect how errors and predictions should be interpreted.

The dataset is included so local training has no network dependency.
Training rejects a changed checksum until the new snapshot is deliberately reviewed.
