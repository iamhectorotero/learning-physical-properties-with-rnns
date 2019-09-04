import unittest
import pandas as pd
import numpy as np

from isaac import feature_engineering
from isaac.constants import MOUSE_POSITION_COLS, MOUSE_CONTROL_COLS, S_PER_FRAME


class TestAddMouseColumnsToPassiveTrials(unittest.TestCase):
    @staticmethod
    def are_columns_in_dataframes(dataframes, columns):
        for df in dataframes:
            for column in columns:
                if column not in df.columns:
                    return False
        return True

    @staticmethod
    def do_columns_have_a_correct_constant_value(dataframes, columns):

        def is_series_constant(series, constant_value=None):
            if len(series) == 0:
                return True

            if constant_value is None or series.iloc[0] == constant_value:
                return (series == series.iloc[0]).all()
            return False

        for df in dataframes:
            for col in columns:
                if col in MOUSE_POSITION_COLS:
                    constant_value = 0.
                elif col == "C_none":
                    constant_value = True
                else:
                    constant_value = False

                if not is_series_constant(df[col], constant_value):
                    return False
        return True

    def test_columns_are_added(self):
        trials = [pd.DataFrame(np.random.rand(5, 10)) for _ in range(7)]

        feature_engineering.add_mouse_columns_to_passive_trials(trials)
        columns = list(MOUSE_POSITION_COLS) + list(MOUSE_CONTROL_COLS)
        self.assertTrue(self.are_columns_in_dataframes(trials, columns))

    def test_constant_value_is_set(self):
        trials = [pd.DataFrame(np.random.rand(5, 10)) for _ in range(7)]

        feature_engineering.add_mouse_columns_to_passive_trials(trials)
        columns = list(MOUSE_POSITION_COLS) + list(MOUSE_CONTROL_COLS)
        self.assertTrue(self.do_columns_have_a_correct_constant_value(trials, columns))

    def test_with_empty_trials(self):
        trials = [pd.DataFrame() for _ in range(5)]
        feature_engineering.add_mouse_columns_to_passive_trials(trials)
        columns = list(MOUSE_POSITION_COLS) + list(MOUSE_CONTROL_COLS)

        self.assertTrue(self.are_columns_in_dataframes(trials, columns))
        self.assertTrue(self.do_columns_have_a_correct_constant_value(trials, columns))

    def test_with_different_length_trials(self):
        trials = []
        for _ in range(8):
            trial_length = np.random.randint(0, 100)
            trial = pd.DataFrame(np.random.rand(trial_length, 10))
            trials.append(trial)

        feature_engineering.add_mouse_columns_to_passive_trials(trials)
        columns = list(MOUSE_POSITION_COLS) + list(MOUSE_CONTROL_COLS)

        self.assertTrue(self.are_columns_in_dataframes(trials, columns))
        self.assertTrue(self.do_columns_have_a_correct_constant_value(trials, columns))

    def test_with_trials_with_mouse_columns(self):
        columns = list(MOUSE_POSITION_COLS) + list(MOUSE_CONTROL_COLS)
        trials = [pd.DataFrame(np.random.rand(5, len(columns)),
                               columns=columns) for _ in range(7)]
        feature_engineering.add_mouse_columns_to_passive_trials(trials)

        self.assertTrue(self.are_columns_in_dataframes(trials, columns))
        self.assertTrue(self.do_columns_have_a_correct_constant_value(trials, columns))


class TestSetMouseVelocities(unittest.TestCase):
    @staticmethod
    def are_mouse_velocity_columns_in_every_trial(trials, columns):
        for trial in trials:
            for column in columns:
                if column not in trial.columns:
                    return False
        return True

    @staticmethod
    def are_velocities_correct(trials):
        def are_velocities_in_axes_correct(pos, vel):
            # Tests velocities are correct by calculating a position from the current on
            # and the velocity
            pos = np.array(pos)
            vel = np.array(vel)

            shifted_pos = pos[1:]
            calculated_pos = (pos[:-1] + vel[1:] * S_PER_FRAME)
            return np.isclose(calculated_pos, shifted_pos).all()

        for trial in trials:
            are_positions_in_x_correct = are_velocities_in_axes_correct(trial["mouseX"],
                                                                        trial["mouse.vx"])

            are_positions_in_y_correct = are_velocities_in_axes_correct(trial["mouseY"],
                                                                        trial["mouse.vy"])
            if not are_positions_in_x_correct or not are_positions_in_y_correct:
                return False

        return True

    def test_moving_mouse(self):
        columns = list(MOUSE_POSITION_COLS)
        trials = [pd.DataFrame(np.random.rand(25, len(columns)),
                               columns=columns) for _ in range(7)]
        feature_engineering.set_mouse_velocities(trials)

        self.assertTrue(self.are_mouse_velocity_columns_in_every_trial(trials, MOUSE_POSITION_COLS))
        self.assertTrue(self.are_velocities_correct(trials))

    def test_static_mouse(self):
        columns = list(MOUSE_POSITION_COLS)
        trials = [pd.DataFrame(np.zeros(50).reshape(25, len(columns)),
                               columns=columns) for _ in range(7)]
        feature_engineering.set_mouse_velocities(trials)

        self.assertTrue(self.are_mouse_velocity_columns_in_every_trial(trials, MOUSE_POSITION_COLS))
        self.assertTrue(self.are_velocities_correct(trials))

    def test_with_different_length_trials(self):
        columns = list(MOUSE_POSITION_COLS)

        trials = []
        for _ in range(8):
            trial_length = np.random.randint(0, 100)
            trial = pd.DataFrame(np.random.rand(trial_length, len(columns)), columns=columns)
            trials.append(trial)

        feature_engineering.set_mouse_velocities(trials)
        self.assertTrue(self.are_mouse_velocity_columns_in_every_trial(trials, MOUSE_POSITION_COLS))
        self.assertTrue(self.are_velocities_correct(trials))


if __name__ == "__main__":
    unittest.main()

