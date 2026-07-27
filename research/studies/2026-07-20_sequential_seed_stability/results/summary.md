# Sequential Seed Stability Results

## Seed-level forced Bijan evidence

- **current, AJ unavailable:** gain mean `-9.34`, seed SD `22.30`, range `[-51.35, +33.16]`; LCB80 positive in `12.5%` of seeds.
- **current, AJ available:** gain mean `-0.77`, seed SD `15.93`, range `[-26.19, +18.96]`; LCB80 positive in `43.8%` of seeds.
- **nested, AJ unavailable:** gain mean `-8.99`, seed SD `20.74`, range `[-46.24, +30.48]`; LCB80 positive in `18.8%` of seeds.
- **nested, AJ available:** gain mean `-8.90`, seed SD `21.07`, range `[-48.05, +30.61]`; LCB80 positive in `18.8%` of seeds.

## AJ-off minus AJ-on within the same root seed

- **current:** correlation `0.25`, mean edge change `-8.58`, SD `23.95`, range `[-35.36, +53.36]`.
- **nested:** correlation `1.00`, mean edge change `-0.09`, SD `1.03`, range `[-2.68, +1.80]`.

## Independent evidence-bank panels (nested AJ-available state)

- **1 bank(s):** panel-gain SD `21.07`, range `[-48.05, +30.61]`; LCB80 positive in `18.8%` of panels.
- **2 bank(s):** panel-gain SD `13.99`, range `[-40.62, +29.48]`; LCB80 positive in `15.0%` of panels.
- **4 bank(s):** panel-gain SD `9.13`, range `[-34.22, +17.82]`; LCB80 positive in `6.9%` of panels.
- **8 bank(s):** panel-gain SD `5.27`, range `[-24.51, +6.71]`; LCB80 positive in `0.6%` of panels.

## Variance decomposition (nested AJ-available state)

- **all_varied:** gain mean `-8.90`, seed SD `21.07`, range `[-48.05, +30.61]`.
- **auction_paths_only:** gain mean `+30.37`, seed SD `3.95`, range `[+23.75, +37.88]`.
- **construction_only:** gain mean `-1.94`, seed SD `16.49`, range `[-25.44, +30.61]`.
- **evidence_only:** gain mean `+26.14`, seed SD `14.43`, range `[-2.52, +53.42]`.
- **weekly_seasons_only:** gain mean `+26.83`, seed SD `12.69`, range `[+6.32, +51.95]`.

The production decision should use a fixed multi-bank, player-keyed evidence panel and add banks adaptively when independent bank estimates disagree or the action boundary remains unresolved. A single root seed should not be treated as a robustness guarantee.
