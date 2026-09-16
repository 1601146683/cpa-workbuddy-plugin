package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestIsContentBlocked(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"business code", 400, `{"code":11128,"msg":"request rejected"}`, true},
		{"string business code", 400, `{"code":"11128"}`, true},
		{"security marker", 400, "The request was blocked by security policy", true},
		{"channel marker", 400, "Illegal API invocation from an unapproved channel", true},
		{"sse error frame", 200, "upstream 200: illegal api invocation", true},
		{"other bad request", 400, `{"code":11101,"msg":"invalid request"}`, false},
		{"server response is not content policy", 500, "blocked by security policy", false},
		{"healthy", 200, "normal response", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isContentBlocked(tt.status, tt.body); got != tt.want {
				t.Fatalf("isContentBlocked(%d, %q) = %v, want %v", tt.status, tt.body, got, tt.want)
			}
		})
	}
}

func TestContentBlockedDoesNotPenalizeOrRotateAccount(t *testing.T) {
	body := `{"code":11128,"msg":"blocked by security policy"}`
	if isAccountFailure(http.StatusBadRequest, body) {
		t.Fatal("content-blocked request must not count as account failure")
	}
	if isAccountFailure(http.StatusForbidden, body) {
		t.Fatal("content-blocked marker must override account-level 403 classification")
	}
	if shouldRotateOnUpstreamErr(http.StatusForbidden, body) {
		t.Fatal("content-blocked request must not rotate accounts")
	}
}

func TestSanitizeBlockedFingerprintsAcrossMessageShapes(t *testing.T) {
	body := []byte(`{
		"model":"glm",
		"messages":[
			{"role":"developer","content":"You are a coding agent running in the Codex CLI, a terminal-based coding assistant."},
			{"role":"user","content":[{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."}]},
			{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{"name":"shell","arguments":"{\"command\":\"echo 11128\"}"}}]},
			{"role":"assistant","function_call":{"name":"legacy","arguments":"{\"text\":\"x-anthropic-billing-header: cc_version=1; cc_entrypoint=cli;\"}"}}
		]
	}`)
	out := rewriteSystemForUpstream(body)
	text := string(out)
	for _, blocked := range []string{
		"official CLI for Claude",
		"running in the Codex CLI, a terminal-based coding assistant",
		"11128",
		"x-anthropic-billing-header",
		"cc_entrypoint=",
	} {
		if strings.Contains(text, blocked) {
			t.Fatalf("blocked fingerprint %q remained in %s", blocked, text)
		}
	}
	var obj map[string]any
	if err := json.Unmarshal(out, &obj); err != nil {
		t.Fatal(err)
	}
	messages := obj["messages"].([]any)
	if role := messages[0].(map[string]any)["role"]; role != "system" {
		t.Fatalf("developer role = %v, want system", role)
	}
}

func TestContentBlockedRetryRewritesSystemOnce(t *testing.T) {
	body := []byte(`{
		"model":"glm",
		"messages":[
			{"role":"system","content":"original system"},
			{"role":"developer","content":"original developer"},
			{"role":"user","content":"keep user"},
			{"role":"assistant","tool_calls":[{"function":{"name":"shell","arguments":"{\"command\":\"keep tool\"}"}}]}
		]
	}`)
	state := &contentBlockedRetryState{}
	next, retry := state.nextBody(body, 400, `upstream 400: {"code":11128}`, false)
	if !retry {
		t.Fatal("first content block must trigger degraded retry")
	}
	text := string(next)
	if strings.Contains(text, "original system") || strings.Contains(text, "original developer") {
		t.Fatalf("original privileged prompts remained: %s", text)
	}
	if !strings.Contains(text, degradedSystemPrompt) || !strings.Contains(text, "keep user") || !strings.Contains(text, "keep tool") {
		t.Fatalf("fallback body did not preserve user/tool content: %s", text)
	}
	if _, retryAgain := state.nextBody(next, 400, "unapproved channel", false); retryAgain {
		t.Fatal("second content block must not trigger another retry")
	}
	rebuiltForNextAccount := state.preserveFallback(body)
	if strings.Contains(string(rebuiltForNextAccount), "original system") || !strings.Contains(string(rebuiltForNextAccount), degradedSystemPrompt) {
		t.Fatalf("fallback mode was not preserved after account rebuild: %s", rebuiltForNextAccount)
	}
}

func TestContentBlockedRetryDoesNotRestartAfterOutput(t *testing.T) {
	state := &contentBlockedRetryState{}
	body := []byte(`{"messages":[{"role":"system","content":"x"}]}`)
	if _, retry := state.nextBody(body, 200, "upstream 200: illegal api invocation", true); retry {
		t.Fatal("must not retry after streamed output reached the client")
	}
	if state.attempted {
		t.Fatal("a skipped retry must not consume the one-shot retry state")
	}
}

func TestRetryContentBlockedRequestKeepsSameAccount(t *testing.T) {
	body := []byte(`{"model":"glm","messages":[{"role":"system","content":"original"},{"role":"user","content":"hello"}]}`)
	req, err := http.NewRequest(http.MethodPost, "https://same-account.example/v2/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer same-account-token")
	req.Header.Set("X-Trace-ID", "trace-1")
	state := &contentBlockedRetryState{}

	rebuilt, retried, err := retryContentBlockedRequest(req, 400, `{"code":11128}`, false, state)
	if err != nil {
		t.Fatal(err)
	}
	if !retried {
		t.Fatal("expected degraded retry request")
	}
	if rebuilt.URL.String() != req.URL.String() {
		t.Fatalf("URL changed from %q to %q; retry must use same account", req.URL, rebuilt.URL)
	}
	if got := rebuilt.Header.Get("Authorization"); got != "Bearer same-account-token" {
		t.Fatalf("authorization changed: %q", got)
	}
	if got := rebuilt.Header.Get("X-Trace-ID"); got != "trace-1" {
		t.Fatalf("non-auth headers not preserved: %q", got)
	}
	if rebuilt.GetBody == nil {
		t.Fatal("rebuilt request must retain GetBody for later account failover")
	}
	bodyRC, err := rebuilt.GetBody()
	if err != nil {
		t.Fatal(err)
	}
	rewrittenBody, err := io.ReadAll(bodyRC)
	_ = bodyRC.Close()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(rewrittenBody), "original") || !strings.Contains(string(rewrittenBody), degradedSystemPrompt) {
		t.Fatalf("unexpected degraded body: %s", rewrittenBody)
	}
	second, retriedAgain, err := retryContentBlockedRequest(rebuilt, 400, "blocked by security policy", false, state)
	if err != nil {
		t.Fatal(err)
	}
	if retriedAgain || second != rebuilt {
		t.Fatal("second content block must return the current request without retry")
	}
}

func TestPumpContentBlockedRetryIgnoresZeroAccountBudget(t *testing.T) {
	oldBudget := loadedRetryOn4xx()
	setRetryOn4xx(0)
	t.Cleanup(func() { setRetryOn4xx(oldBudget) })

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"code":11128,"msg":"blocked by security policy"}`))
	}))
	t.Cleanup(server.Close)

	body := []byte(`{"model":"glm","messages":[{"role":"system","content":"blocked prompt"},{"role":"user","content":"hello"}]}`)
	req, err := http.NewRequest(http.MethodPost, server.URL, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	pumpUpstreamStream(req, nil, "", false, "glm", "glm", "", time.Now(), "", "", "", "")
	if got := calls.Load(); got != 2 {
		t.Fatalf("upstream calls = %d, want 2 (original plus same-account degraded retry with account budget 0)", got)
	}
}
