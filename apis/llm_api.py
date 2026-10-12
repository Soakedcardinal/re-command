import requests
import json
import sys
import re
import time
import os

# Built-in prompt template; {listening_history} is replaced with the scrobble JSON.
DEFAULT_PROMPT_TEMPLATE = """You are a music expert assistant. Based on the following list of recently listened tracks in JSON format, recommend 25 new songs that this listener might like; only include tracks that are not already in the listening history.
The recommendations should be for a user who enjoys the artists and genres represented in the listening history.

My listening history:
{listening_history}

Only respond with a single JSON array of objects, where each object represents a recommended track and has the keys "artist", "title", and "album".

Example response format:
[
  {"artist": "Example Artist 1", "title": "Example Song 1", "album": "Example Album 1"},
  {"artist": "Example Artist 2", "title": "Example Song 2", "album": "Example Album 2"}
]
"""

class LlmCancelled(Exception):
    """Raised when a streaming LLM generation is cancelled by the caller."""


class LlmAPI:
    def __init__(self, provider, gemini_api_key=None, openrouter_api_key=None, llama_api_key=None, model_name=None, base_url=None, llama_max_tokens=None, prompt_template=None):
        self.provider = provider
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key
        self.llama_api_key = llama_api_key
        self.model_name = model_name
        self.base_url = base_url
        self.llama_max_tokens = llama_max_tokens
        self.prompt_template = prompt_template
        self.last_stats = None

        if self.provider == 'gemini' and self.gemini_api_key:
            # We use the REST API directly for Gemini to avoid version conflicts in the SDK
            # and to ensure compatibility with newer models like gemini-3-flash-preview
            self.gemini_model = self.model_name or 'gemini-2.5-flash'
            # Ensure model name is prefixed with models/ if not already
            if not self.gemini_model.startswith('models/'):
                full_model_path = f"models/{self.gemini_model}"
            else:
                full_model_path = self.gemini_model
            
            self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/{full_model_path}:generateContent?key={self.gemini_api_key}"
            self.headers = {"Content-Type": "application/json"}
        elif self.provider == 'openrouter' and self.openrouter_api_key:
            # Use custom base URL if provided, otherwise use OpenRouter's default
            self.openrouter_url = self.base_url or "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
        elif self.provider == 'llama' and self.base_url:
            # For Llama.cpp, use the base URL as the endpoint
            self.llama_url = self.base_url
            self.headers = {
                "Content-Type": "application/json"
            }
            # Add API key if provided (some Llama.cpp servers might require it)
            if self.llama_api_key:
                self.headers["Authorization"] = f"Bearer {self.llama_api_key}"
        else:
            raise ValueError("LLM provider is not configured correctly. Please provide API keys and base URL for Llama.cpp.")

    def _build_prompt(self, scrobbles_json):
        """Builds the prompt for the LLM from the configured template."""
        template = self.prompt_template or DEFAULT_PROMPT_TEMPLATE
        if '{listening_history}' in template:
            return template.replace('{listening_history}', scrobbles_json)
        return f"{template}\n\nMy listening history:\n{scrobbles_json}\n"

    def get_recommendations(self, scrobbles, cancel_event=None):
        """
        Gets music recommendations from the configured LLM provider.
        'scrobbles' is a list of dicts with 'artist' and 'track'.
        'cancel_event' is an optional threading.Event; setting it aborts the request.
        """
        if not scrobbles:
            return []

        scrobbles_json = json.dumps(scrobbles, indent=2)
        prompt = self._build_prompt(scrobbles_json)

        try:
            if self.provider == 'gemini':
                data = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                }
                api_response = requests.post(self.gemini_url, headers=self.headers, json=data)
                if api_response.status_code != 200:
                    print(f"Gemini API Error: {api_response.status_code} {api_response.text}", file=sys.stderr)
                api_response.raise_for_status()
                response_json = api_response.json()
                # Extract text from the response
                try:
                    response_text = response_json['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError):
                    print(f"Gemini API Error: Unexpected response structure: {response_json}", file=sys.stderr)
                    return []
            elif self.provider == 'openrouter':
                openrouter_model = self.model_name or "tngtech/deepseek-r1t2-chimera:free"
                data = {
                    "model": openrouter_model,
                    "messages": [{"role": "user", "content": prompt}]
                }
                api_response = requests.post(self.openrouter_url, headers=self.headers, json=data)
                api_response.raise_for_status()
                response_text = api_response.json()['choices'][0]['message']['content']
            elif self.provider == 'llama':
                # For Llama.cpp, use the OpenAI-compatible API format
                llama_model = self.model_name or "local-model"
                data = {
                    "model": llama_model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.llama_max_tokens:
                    data["max_tokens"] = self.llama_max_tokens
                data["stream"] = True
                self.last_stats = {'prompt_chars': len(prompt)}
                print(f"LLM request: prompt_chars={len(prompt)} "
                      f"max_tokens={'omitted (server window)' if not self.llama_max_tokens else self.llama_max_tokens} "
                      f"model={llama_model} url={self.llama_url}", file=sys.stderr)
                print(f"LLM prompt head: {prompt[:400]!r}", file=sys.stderr)
                # Optional debug capture (RECOMMAND_LLM_DEBUG)
                debug_dump = os.getenv("RECOMMAND_LLM_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
                if debug_dump:
                    try:
                        with open('/tmp/llm_last_prompt.txt', 'w') as f:
                            f.write(prompt)
                    except Exception as e:
                        print(f"LLM Debug: could not write /tmp/llm_last_prompt.txt: {e}", file=sys.stderr)
                t0 = time.time()
                try:
                    api_response = requests.post(self.llama_url, headers=self.headers, json=data,
                                                 stream=True, timeout=(10, None))
                except requests.RequestException as e:
                    if cancel_event is not None and cancel_event.is_set():
                        latency = time.time() - t0
                        print(f"LLM generation cancelled after {latency:.1f}s (connect phase)", file=sys.stderr)
                        self.last_stats = {'latency_s': round(latency, 1), 'completion_tokens': 0,
                                           'finish_reason': 'cancelled', 'reasoning_chars': 0,
                                           'content_chars': 0, 'prompt_chars': len(prompt)}
                        raise LlmCancelled(f"cancelled by request: {e}") from e
                    raise
                if api_response.status_code != 200:
                    print(f"LLM API Error: {api_response.status_code} {api_response.text}", file=sys.stderr)
                api_response.raise_for_status()
                reasoning_parts = []
                content_parts = []
                finish_reason = None
                chunk_count = 0
                for line in api_response.iter_lines(decode_unicode=True):
                    if cancel_event is not None and cancel_event.is_set():
                        latency = time.time() - t0
                        api_response.close()
                        print(f"LLM generation cancelled after {latency:.1f}s "
                              f"({chunk_count} chunk(s), reasoning={len(''.join(reasoning_parts))} chars)", file=sys.stderr)
                        self.last_stats = {
                            'latency_s': round(latency, 1),
                            'completion_tokens': chunk_count,
                            'finish_reason': 'cancelled',
                            'reasoning_chars': len(''.join(reasoning_parts)),
                            'content_chars': len(''.join(content_parts)),
                            'prompt_chars': len(prompt),
                        }
                        if debug_dump:
                            try:
                                with open('/tmp/llm_last_response.txt', 'w') as f:
                                    json.dump({'finish_reason': 'cancelled',
                                               'reasoning': ''.join(reasoning_parts),
                                               'content': ''.join(content_parts)}, f, indent=2)
                            except Exception as e:
                                print(f"LLM Debug: could not write /tmp/llm_last_response.txt: {e}", file=sys.stderr)
                        partial_reasoning = ''.join(reasoning_parts)
                        if partial_reasoning:
                            print(f"LLM reasoning tail: {partial_reasoning[-500:]!r}", file=sys.stderr)
                        partial_content = ''.join(content_parts)
                        if partial_content:
                            print(f"LLM content tail: {partial_content[-500:]!r}", file=sys.stderr)
                        raise LlmCancelled("cancelled by request")
                    if not line or not line.startswith('data:'):
                        continue
                    payload = line[len('data:'):].strip()
                    if payload == '[DONE]':
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get('choices') or []
                    if choices:
                        delta = choices[0].get('delta') or {}
                        if delta.get('content'):
                            content_parts.append(delta['content'])
                        reasoning_delta = delta.get('reasoning_content') or delta.get('reasoning')
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
                        if choices[0].get('finish_reason'):
                            finish_reason = choices[0]['finish_reason']
                        chunk_count += 1
                latency = time.time() - t0
                reasoning = ''.join(reasoning_parts)
                response_text = ''.join(content_parts)
                print(f"LLM response: latency={latency:.1f}s "
                      f"completion_chunks={chunk_count} "
                      f"finish_reason={finish_reason} "
                      f"reasoning_chars={len(reasoning)} "
                      f"content_chars={len(response_text or '')}", file=sys.stderr)
                self.last_stats = {
                    'latency_s': round(latency, 1),
                    'completion_tokens': chunk_count,
                    'finish_reason': finish_reason,
                    'reasoning_chars': len(reasoning),
                    'content_chars': len(response_text or ''),
                    'prompt_chars': len(prompt),
                }
                if debug_dump:
                    try:
                        with open('/tmp/llm_last_response.txt', 'w') as f:
                            json.dump({'finish_reason': finish_reason,
                                       'reasoning': reasoning,
                                       'content': response_text}, f, indent=2)
                    except Exception as e:
                        print(f"LLM Debug: could not write /tmp/llm_last_response.txt: {e}", file=sys.stderr)
                if reasoning:
                    print(f"LLM reasoning head: {reasoning[:300]!r}", file=sys.stderr)
                if not response_text:
                    print(f"LLM API Error: model returned no content "
                          f"(finish_reason={finish_reason}, reasoning={len(reasoning)} chars)",
                          file=sys.stderr)
                    return []
                print(f"LLM content head: {response_text[:300]!r}", file=sys.stderr)
            else:
                return []

            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                print(f"LLM API Error: Could not find a JSON array in the response.\nLLM Raw Response: {response_text}", file=sys.stderr)
                return []
            
            recommendations = json.loads(json_match.group(0))

            # Normalize keys to handle variations from different LLM models
            normalized_recommendations = []
            for rec in recommendations:
                if isinstance(rec, dict):
                    normalized_rec = {}
                    # Map common variations to standard keys
                    key_mappings = {
                        'artist': ['artist', 'artist_name'],
                        'title': ['title', 'song', 'track', 'name'],
                        'album': ['album', 'album_name', 'album_title']
                    }

                    for standard_key, possible_keys in key_mappings.items():
                        for possible_key in possible_keys:
                            if possible_key in rec:
                                normalized_rec[standard_key] = rec[possible_key]
                                break
                        # If no mapping found, set to empty string
                        if standard_key not in normalized_rec:
                            normalized_rec[standard_key] = ''

                    normalized_recommendations.append(normalized_rec)
                else:
                    # If not a dict, skip this recommendation
                    continue

            return normalized_recommendations
        except LlmCancelled:
            raise
        except Exception as e:
            print(f"Error getting recommendations from {self.provider}: {e}", file=sys.stderr)
            return []
