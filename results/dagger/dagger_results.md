## DAgger: network driving on its own (evaluation runs)

| Model | Lane | Steering push | Distance | Lane offset RMS | Max offset | Interventions per 100 m | Expert share | Network vs expert |
|---|---|---|---|---|---|---|---|---|
| lane_mt_v1.pt | ours | none | 73.6 m | 0.7 cm | 3.5 cm | 0.0 | 0 % | 0.0127 rad/s |
| lane_mt_v1.pt | other | none | 73.5 m | 0.95 cm | 2.7 cm | 0.0 | 0 % | 0.0107 rad/s |
| lane_mt_v1.pt | ours | 0.3 rad/s | 73.4 m | 13.62 cm | 30.1 cm | 10.9 | 20 % | 0.0176 rad/s |
| lane_mt_d1.pt | ours | none | 73.6 m | 1.97 cm | 4.2 cm | 0.0 | 0 % | 0.0143 rad/s |
| lane_mt_d1.pt | other | none | 73.5 m | 0.91 cm | 3.3 cm | 0.0 | 0 % | 0.012 rad/s |
| lane_mt_d1.pt | ours | 0.3 rad/s | 73.4 m | 13.54 cm | 29.4 cm | 10.9 | 19 % | 0.021 rad/s |
| lane_mt_d2.pt | ours | none | 73.5 m | 1.9 cm | 4.1 cm | 0.0 | 0 % | 0.0141 rad/s |
| lane_mt_d2.pt | other | none | 73.5 m | 0.93 cm | 3.5 cm | 0.0 | 0 % | 0.0113 rad/s |
| lane_mt_d2.pt | ours | 0.3 rad/s | 73.4 m | 13.93 cm | 29.2 cm | 12.26 | 20 % | 0.0218 rad/s |
| expert (beta 1) | ours | 0.3 rad/s | 73.4 m | 14.13 cm | 30.2 cm | 10.9 | 19 % | 0.0204 rad/s |

## Data collection rounds

| Round | Model driving | beta (expert share of the command) | Lane | Images | Interventions | Lane offset RMS |
|---|---|---|---|---|---|---|
| round1_ours | lane_mt_v1.pt | 0.5 | ours | 1000 | 0 | 0.42 cm |
| round1_other | lane_mt_v1.pt | 0.5 | other | 1000 | 0 | 0.46 cm |
| round2_ours | lane_mt_d1.pt | 0.0 | ours | 1000 | 0 | 1.64 cm |
| round2_other | lane_mt_d1.pt | 0.0 | other | 1000 | 0 | 1.05 cm |
