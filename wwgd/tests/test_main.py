from unittest import TestCase

import wwgd

class TestMain(TestCase):
    def test_completes(self):
        r = wwgd.main()
        self.assertEquals(r, 0)
