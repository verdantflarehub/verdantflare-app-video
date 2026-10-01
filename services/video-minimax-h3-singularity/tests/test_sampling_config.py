import unittest

try:
    from h3_singularity.engine import Engine, RuntimeErrorCode
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise unittest.SkipTest("PyTorch is required for sampling configuration tests") from error
    raise


class HrSamplingConfigTest(unittest.TestCase):
    def test_supported_hr_sampler_scheduler_pairs(self):
        for sampler in ("euler", "er_sde"):
            for scheduler in ("simple", "beta"):
                Engine._validate_hr_sampling_config(sampler, scheduler)

    def test_unknown_hr_sampler_is_rejected(self):
        with self.assertRaisesRegex(RuntimeErrorCode, "invalid_hr_sampler"):
            Engine._validate_hr_sampling_config("ddim", "simple")

    def test_unknown_hr_scheduler_is_rejected(self):
        with self.assertRaisesRegex(RuntimeErrorCode, "invalid_hr_scheduler"):
            Engine._validate_hr_sampling_config("er_sde", "karras")


if __name__ == "__main__":
    unittest.main()
