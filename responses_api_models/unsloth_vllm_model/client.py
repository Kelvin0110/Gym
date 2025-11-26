import asyncio
import time
import aiohttp
import requests


class UnslothVLLMClient:
    def __init__(self, base_url: str = "http://localhost:9800"):
        self.base_url = base_url
        self.responses_url = f"{base_url}/v1/responses"
        self.sleep_url = f"{base_url}/v1/engine/sleep"
        self.status_url = f"{base_url}/v1/engine/status"

    def create_request(self, prompt: str, max_output_tokens: int = 100) -> dict:
        return {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                    "type": "message",
                }
            ],
            "max_output_tokens": max_output_tokens,
            "temperature": 1.0,
        }

    def test_single_request(self, prompt: str = "What is 2+2?"):
        print("\n" + "=" * 80)
        print("TEST: Single Request")
        print("=" * 80)

        request = self.create_request(prompt)
        print(f"Prompt: {prompt}")

        start = time.time()
        response = requests.post(self.responses_url, json=request)
        elapsed = time.time() - start

        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")

        if response.status_code == 200:
            data = response.json()
            output = data["output"][-1]["content"][0]["text"]
            print(f"Response: {output}")

            # Check for token information
            if "generation_token_ids" in data["output"][-1]:
                token_ids = data["output"][-1]["generation_token_ids"]
                log_probs = data["output"][-1]["generation_log_probs"]
                print(f"Token IDs: {len(token_ids)} tokens")
                print(f"Log Probs: {len(log_probs)} values")
                print(f"Sample token IDs: {token_ids[:5]}...")
                print(f"Sample log probs: {[f'{lp:.3f}' for lp in log_probs[:5]]}...")
            else:
                print("⚠️  Warning: No token information returned")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    async def test_batched_requests(self, num_requests: int = 4):
        print("\n" + "=" * 80)
        print(f"TEST: Batched Requests (n={num_requests})")
        print("=" * 80)

        prompts = [
            f"Count from 1 to {i+1}" for i in range(num_requests)
        ]

        print(f"Sending {num_requests} concurrent requests...")

        start = time.time()

        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in prompts:
                request = self.create_request(prompt, max_output_tokens=50)
                task = session.post(self.responses_url, json=request)
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

        elapsed = time.time() - start

        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {num_requests / elapsed:.2f} requests/sec")

        success_count = 0
        for i, response in enumerate(responses):
            if response.status == 200:
                data = await response.json()
                output = data["output"][-1]["content"][0]["text"]
                print(f"  [{i+1}] {prompts[i][:30]}... → {output[:50]}...")
                success_count += 1
            else:
                print(f"  [{i+1}] Error: {response.status}")

        print(f"\nSuccess rate: {success_count}/{num_requests}")
        return success_count == num_requests

    def test_engine_status(self):
        """Test engine status endpoint."""
        print("\n" + "=" * 80)
        print("TEST: Engine Status")
        print("=" * 80)

        response = requests.get(self.status_url)
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"VRAM Usage: {data['vram_usage_gb']:.2f} GB")
            print(f"Batch Queue Size: {data['batch_queue_size']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    def test_sleep_wake(self):
        """Test sleep/wake functionality."""
        print("\n" + "=" * 80)
        print("TEST: Sleep/Wake")
        print("=" * 80)

        # Get initial status
        status1 = requests.get(self.status_url).json()
        print(f"Initial VRAM: {status1['vram_usage_gb']:.2f} GB")

        # Sleep
        print("Putting engine to sleep...")
        sleep_response = requests.post(self.sleep_url)
        if sleep_response.status_code != 200:
            print(f"Sleep failed: {sleep_response.text}")
            return False

        print(f"Sleep status: {sleep_response.json()['status']}")

        # Check status after sleep
        status2 = requests.get(self.status_url).json()
        print(f"After sleep VRAM: {status2['vram_usage_gb']:.2f} GB")
        print(f"VRAM freed: {status1['vram_usage_gb'] - status2['vram_usage_gb']:.2f} GB")

        # Wake by sending a request
        print("\nWaking engine with request...")
        wake_success = self.test_single_request("What is 1+1?")

        # Check status after wake
        status3 = requests.get(self.status_url).json()
        print(f"After wake status: {status3['status']}")
        print(f"After wake VRAM: {status3['vram_usage_gb']:.2f} GB")

        return wake_success

    def test_reasoning_parsing(self):
        print("\n" + "=" * 80)
        print("TEST: Reasoning Parsing")
        print("=" * 80)

        prompt = "Solve step by step: What is 15 * 23?"

        request = self.create_request(prompt, max_output_tokens=200)
        response = requests.post(self.responses_url, json=request)

        if response.status_code == 200:
            data = response.json()
            output_items = data["output"]

            print(f"Output items: {len(output_items)}")

            reasoning_items = [item for item in output_items if item.get("type") == "reasoning"]
            message_items = [item for item in output_items if item.get("type") == "message"]

            print(f"Reasoning items: {len(reasoning_items)}")
            print(f"Message items: {len(message_items)}")

            if reasoning_items:
                print("\nReasoning content:")
                for item in reasoning_items:
                    for summary in item.get("summary", []):
                        print(f"  {summary['text'][:100]}...")

            if message_items:
                print("\nFinal answer:")
                print(f"  {message_items[0]['content'][0]['text']}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    def run_all_tests(self):
        print("\n" + "=" * 80)
        print("Running all tests...")
        print(f"Server: {self.base_url}")

        results = {}

        try:
            results["single_request"] = self.test_single_request()
        except Exception as e:
            print(f"Single request test failed: {e}")
            results["single_request"] = False

        try:
            results["batched_requests"] = asyncio.run(self.test_batched_requests(4))
        except Exception as e:
            print(f"Batched requests test failed: {e}")
            results["batched_requests"] = False

        try:
            results["engine_status"] = self.test_engine_status()
        except Exception as e:
            print(f"Engine status test failed: {e}")
            results["engine_status"] = False

        try:
            results["sleep_wake"] = self.test_sleep_wake()
        except Exception as e:
            print(f"Sleep/wake test failed: {e}")
            results["sleep_wake"] = False

        try:
            results["reasoning_parsing"] = self.test_reasoning_parsing()
        except Exception as e:
            print(f"Reasoning parsing test failed: {e}")
            results["reasoning_parsing"] = False

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        for test, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{status}: {test}")

        total = len(results)
        passed = sum(results.values())
        print(f"\nTotal: {passed}/{total} tests passed")

        return passed == total


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Unsloth-vLLM Model Server")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:9800",
        help="Server URL (default: http://localhost:9800)",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "single",
            "batch",
            "status",
            "sleep",
            "reasoning",
            "all",
        ],
        default="all",
        help="Test to run (default: all)",
    )

    args = parser.parse_args()

    client = UnslothVLLMClient(base_url=args.url)

    if args.test == "single":
        client.test_single_request()
    elif args.test == "batch":
        asyncio.run(client.test_batched_requests())
    elif args.test == "status":
        client.test_engine_status()
    elif args.test == "sleep":
        client.test_sleep_wake()
    elif args.test == "reasoning":
        client.test_reasoning_parsing()
    else:
        client.run_all_tests()


if __name__ == "__main__":
    main()
