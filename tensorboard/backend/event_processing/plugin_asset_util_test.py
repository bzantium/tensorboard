import unittest
from unittest import mock

from tensorboard.backend.event_processing import plugin_asset_util


class PluginAssetUtilTest(unittest.TestCase):
    def test_list_plugins_unsupported_filesystem_returns_empty(self):
        with mock.patch.object(
            plugin_asset_util.tf.io.gfile,
            "listdir",
            side_effect=ValueError("No recognized filesystem for prefix gs"),
        ):
            self.assertEqual(
                [], plugin_asset_util.ListPlugins("gs://bucket/run")
            )

    def test_list_assets_unsupported_filesystem_returns_empty(self):
        with mock.patch.object(
            plugin_asset_util.tf.io.gfile,
            "listdir",
            side_effect=ValueError("No recognized filesystem for prefix gs"),
        ):
            self.assertEqual(
                [],
                plugin_asset_util.ListAssets("gs://bucket/run", "projector"),
            )

    def test_retrieve_asset_unsupported_filesystem_raises_key_error(self):
        with mock.patch.object(
            plugin_asset_util.tf.io.gfile,
            "GFile",
            side_effect=ValueError("No recognized filesystem for prefix gs"),
        ):
            with self.assertRaisesRegex(KeyError, "Asset path"):
                plugin_asset_util.RetrieveAsset(
                    "gs://bucket/run", "projector", "projector_config.pbtxt"
                )


if __name__ == "__main__":
    unittest.main()
