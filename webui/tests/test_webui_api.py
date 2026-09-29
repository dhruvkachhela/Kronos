import unittest
import urllib.request
import json

class TestKronosWebUI(unittest.TestCase):
    BASE_URL = "http://localhost:7070"

    def test_available_models(self):
        url = f"{self.BASE_URL}/api/available-models"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("model_available"))
            self.assertIn("kronos-mini", data.get("models", {}))

    def test_data_files(self):
        url = f"{self.BASE_URL}/api/data-files"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(data, list)
            self.assertTrue(len(data) > 0)

    def test_model_status(self):
        url = f"{self.BASE_URL}/api/model-status"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("available"))
            self.assertTrue(data.get("loaded"))

if __name__ == "__main__":
    unittest.main()
