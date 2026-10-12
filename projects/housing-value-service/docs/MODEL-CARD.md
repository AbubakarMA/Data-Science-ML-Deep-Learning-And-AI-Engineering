# Model card

## Intended use

Demonstrate and understand an end-to-end regression service using historical
California census districts. Users supply district-level characteristics and
receive an estimated historical district median house value in US dollars.

This model has not been evaluated for current property valuations, Ghanaian
housing markets, individual homes or consequential lending decisions.

## Data and evaluation

The 20,640 records are split deterministically into 12,384 training records,
4,128 validation records and 4,128 final test records using seed 42.
Preprocessing statistics and categories are fitted exclusively on training records.

The fixed release gate requires validation RMSE no more than 80% of the baseline's
RMSE and validation MAE no more than 90% of the baseline's MAE. The candidate
passed before final test reporting.

The deployed candidate is the NumPy implementation, not the library comparison.
Test MAE is about USD 47,642; test RMSE is about USD 65,523; test R squared is
about 0.681. The mean baseline's test MAE is about USD 91,442.

## Assumptions and limitations

- Districts are not independent in space. A random split can put nearby districts
  in different partitions. It is not evidence of performance in a new region.
- The data is historical. The recorded relationships do not establish today's prices.
- Targets and housing age are capped; linear predictions do not recover values
  hidden by censoring.
- A linear model can miss nonlinear relationships and interactions.
- Clipping reduces the effect of extreme numerical features but also loses information.
- The ISLAND category has very few examples; its behavior is weakly supported.
- An error metric across the test set is not an uncertainty interval for one input.
- Feature contributions describe arithmetic, not causal effects.

Before broad use, evaluate spatial and temporal holdouts, error across supported
district groups and new data quality. Obtain labels from the intended deployment
population, establish uncertainty estimates and compare suitable challengers.
