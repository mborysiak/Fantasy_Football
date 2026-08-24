import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from Scripts.Modeling.publish_actual_salaries import (
    ACTUAL_METHOD_VERSION,
    SALARY_RESID_COLUMNS,
    build_actual_salary_slice,
    publish_actual_salary_slice,
)


class ActualSalaryPublisherTest(unittest.TestCase):

    def _build_fixture(self, database: Path) -> None:
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                '''CREATE TABLE Actual_Salaries (
                       player TEXT,
                       actual_salary REAL,
                       year INTEGER,
                       league TEXT
                   )'''
            )
            connection.executemany(
                'INSERT INTO Actual_Salaries VALUES (?, ?, 2026, ?)',
                [
                    ('Alias One', 17.0, 'nv'),
                    ('Alias Two', 0.0, 'nv'),
                    ('Kicker', 1.0, 'nv'),
                ],
            )
            connection.commit()
            connection.execute(
                '''CREATE TABLE Final_Predictions_Resid (
                       player_key TEXT,
                       player TEXT,
                       pos TEXT,
                       year INTEGER,
                       version TEXT,
                       dataset TEXT
                   )'''
            )
            connection.executemany(
                '''INSERT INTO Final_Predictions_Resid
                   VALUES (?, ?, ?, 2026, 'nv', 'final_ensemble')''',
                [
                    ('key-one', 'Player One', 'QB'),
                    ('key-two', 'Player Two', 'TE'),
                ],
            )
            connection.commit()

    def test_build_filters_to_canonical_pool_and_zeroes_uncertainty(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / 'Simulation.sqlite3'
            self._build_fixture(database)

            def resolve(rows, *_args, **_kwargs):
                output = rows.copy()
                output['player_key'] = ['key-one', 'key-two', pd.NA]
                output['eligibility_key_match_method'] = [
                    'alias_confirmed_unique',
                    'alias_confirmed_unique',
                    'unresolved',
                ]
                return output

            with (
                patch(
                    'Scripts.Modeling.publish_actual_salaries.load_identity_frames',
                    return_value=(pd.DataFrame(), pd.DataFrame()),
                ),
                patch(
                    'Scripts.Modeling.publish_actual_salaries.resolve_source_player_keys',
                    side_effect=resolve,
                ),
            ):
                output = build_actual_salary_slice(
                    database,
                    Path(directory) / 'Projection_V2_nv.sqlite3',
                    year=2026,
                    league='nv',
                    expected_pool_rows=2,
                )

            self.assertEqual(output.player.tolist(), ['Player One', 'Player Two'])
            self.assertEqual(output.salary.tolist(), [17.0, 0.0])
            self.assertEqual(output.league.unique().tolist(), ['nv_actual'])
            self.assertTrue((output.std_dev == 0).all())
            self.assertTrue(output.min_score.equals(output.salary))
            self.assertTrue(output.max_score.equals(output.salary))
            self.assertTrue((output[list(SALARY_RESID_COLUMNS)] == 0).all().all())
            self.assertEqual(
                output.salary_method_version.unique().tolist(),
                [ACTUAL_METHOD_VERSION],
            )

    def test_publish_replaces_only_actual_slice_and_creates_backup(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / 'Simulation.sqlite3'
            self._build_fixture(database)

            def resolve(rows, *_args, **_kwargs):
                output = rows.copy()
                output['player_key'] = ['key-one', 'key-two', pd.NA]
                output['eligibility_key_match_method'] = [
                    'alias_confirmed_unique',
                    'alias_confirmed_unique',
                    'unresolved',
                ]
                return output

            with (
                patch(
                    'Scripts.Modeling.publish_actual_salaries.load_identity_frames',
                    return_value=(pd.DataFrame(), pd.DataFrame()),
                ),
                patch(
                    'Scripts.Modeling.publish_actual_salaries.resolve_source_player_keys',
                    side_effect=resolve,
                ),
            ):
                output = build_actual_salary_slice(
                    database,
                    Path(directory) / 'Projection_V2_nv.sqlite3',
                    year=2026,
                    league='nv',
                    expected_pool_rows=2,
                )

            with closing(sqlite3.connect(database)) as connection:
                output.head(0).to_sql(
                    'Salaries_Pred',
                    connection,
                    if_exists='replace',
                    index=False,
                )
                predicted = output.iloc[[0]].copy()
                predicted['league'] = 'nvpred'
                predicted.to_sql(
                    'Salaries_Pred',
                    connection,
                    if_exists='append',
                    index=False,
                )
                connection.commit()

            backup = publish_actual_salary_slice(output, database)

            self.assertTrue(backup.exists())
            with closing(sqlite3.connect(database)) as connection:
                counts = dict(connection.execute(
                    '''SELECT league, COUNT(*)
                       FROM Salaries_Pred
                       GROUP BY league'''
                ))
            self.assertEqual(counts, {'nv_actual': 2, 'nvpred': 1})


if __name__ == '__main__':
    unittest.main()
