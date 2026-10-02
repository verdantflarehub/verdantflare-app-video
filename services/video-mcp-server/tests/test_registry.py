from __future__ import annotations

import asyncio
import base64
import json
import unittest

import httpx

from app.registry import EtcdRegistry


class TestEtcdRegistry(unittest.IsolatedAsyncioTestCase):
    async def test_metadata_structure(self) -> None:
        tools = [
            {
                "name": "video.create",
                "description": "Create video task",
                "inputSchema": {"type": "object"},
            }
        ]
        registry = EtcdRegistry(
            domain="video",
            tools=tools,
            endpoint="http://video-mcp-server:8000/",
            health_endpoint="http://video-mcp-server:8000/health",
            version="v1.0.0",
        )
        meta = registry.build_metadata()
        self.assertEqual(meta["domain"], "video")
        self.assertEqual(meta["endpoint"], "http://video-mcp-server:8000/")
        self.assertEqual(meta["health_endpoint"], "http://video-mcp-server:8000/health")
        self.assertEqual(meta["version"], "v1.0.0")
        self.assertIn("updated_at", meta)
        self.assertEqual(len(meta["tools"]), 1)
        self.assertEqual(meta["tools"][0]["name"], "video.create")

    async def test_registration_lifecycle(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            body = json.loads(request.content) if request.content else {}

            if request.url.path == "/v3/lease/grant":
                self.assertEqual(body.get("TTL"), 300)
                return httpx.Response(200, json={"ID": "123456789", "TTL": "300"})
            elif request.url.path == "/v3/kv/put":
                self.assertEqual(body.get("lease"), "123456789")
                key = base64.b64decode(body["key"]).decode("utf-8")
                self.assertEqual(key, "/verdantflare/mcp/services/video")
                val = json.loads(base64.b64decode(body["value"]).decode("utf-8"))
                self.assertEqual(val["domain"], "video")
                return httpx.Response(200, json={"header": {"revision": "1"}})
            elif request.url.path == "/v3/lease/keepalive":
                self.assertEqual(body.get("ID"), "123456789")
                return httpx.Response(200, json={"result": {"ID": "123456789", "TTL": "300"}})
            elif request.url.path == "/v3/lease/revoke":
                self.assertEqual(body.get("ID"), "123456789")
                return httpx.Response(200, json={"header": {"revision": "2"}})
            return httpx.Response(404)

        registry = EtcdRegistry(
            domain="video",
            tools=[{"name": "video.create", "description": "d", "inputSchema": {}}],
            etcd_endpoints="http://mock-etcd:2379",
        )
        registry._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        # 1. Register
        ok = await registry.register()
        self.assertTrue(ok)
        self.assertEqual(registry._lease_id, "123456789")
        self.assertIn("/v3/lease/grant", calls)
        self.assertIn("/v3/kv/put", calls)

        # 2. Keepalive
        alive = await registry.keepalive()
        self.assertTrue(alive)
        self.assertIn("/v3/lease/keepalive", calls)

        # 3. Deregister
        await registry.deregister()
        self.assertIsNone(registry._lease_id)
        self.assertIn("/v3/lease/revoke", calls)

        await registry.stop()


if __name__ == "__main__":
    unittest.main()
