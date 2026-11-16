/* Compile against the patched native source; no camera or USB access. */
#include "ustreamer/http/server.h"
static void test_reply(struct evhttp_request *, int, const char *, struct evbuffer *);
#define evhttp_send_reply test_reply
#include "ustreamer/http/server.c"
#undef evhttp_send_reply

static int replies;
static void test_reply(struct evhttp_request *request, int code, const char *reason,
                       struct evbuffer *body) {
    (void)request; (void)reason;
    assert(code == 200);
    assert(evbuffer_get_length(body) == 4);
    assert(memcmp(evbuffer_pullup(body, -1), "JPEG", 4) == 0);
    ++replies;
}
static const char *header(struct evhttp_request *req, const char *key) {
    return evhttp_find_header(evhttp_request_get_output_headers(req), key);
}
static struct evhttp_request *snapshot(server_s *server) {
    struct evhttp_request *req = evhttp_request_new(NULL, NULL);
    assert(req != NULL);
    _http_callback_snapshot(req, server);
    assert(strcmp(header(req, "Content-Type"), "image/jpeg") == 0);
    assert(header(req, "X-Timestamp") != NULL);
    assert(header(req, "X-UStreamer-Grab-Timestamp") != NULL);
    assert(header(req, "X-UStreamer-Send-Timestamp") != NULL);
    return req;
}
int main(void) {
    frame_s *raw = frame_init(), *jpeg = frame_init(), *out = frame_init();
    raw->width = 1280; raw->height = 720; raw->online = true;
    raw->format = V4L2_PIX_FMT_MJPEG;
    raw->capture_timestamp_ns = 9876543210123ULL;
    raw->capture_epoch_ns = 123456789ULL;
    raw->capture_sequence = UINT32_MAX;
    raw->capture_flags = V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_SOE;
    raw->grab_ts = 9876.60L;
    frame_set_data(raw, (const uint8_t *)"JPEG", 4);
    frame_copy_meta(raw, jpeg); /* encoder metadata path */
    assert(jpeg->capture_sequence == UINT32_MAX);
    frame_copy(raw, out); /* MJPEG passthrough and exposure copy path */
    assert(out->capture_timestamp_ns == raw->capture_timestamp_ns);
    assert(out->capture_epoch_ns == raw->capture_epoch_ns);
    assert(out->capture_flags == raw->capture_flags);
    assert(out->capture_sequence == raw->capture_sequence);
    assert(out->grab_ts == raw->grab_ts);
    exposed_s exposed = {.frame = out};
    server_runtime_s run = {.exposed = &exposed};
    server_s server = {.run = &run, .allow_origin = ""};
    for (int i = 0; i < 2; ++i) {
        struct evhttp_request *req = snapshot(&server);
        assert(strcmp(header(req, "X-Capture-Monotonic-Ns"), "9876543210123") == 0);
        assert(strcmp(header(req, "X-Capture-Stream-Ns"), "123456789") == 0);
        assert(strcmp(header(req, "X-Capture-Sequence"), "4294967295") == 0);
        assert(strcmp(header(req, "X-Capture-V4L2-Flags"), "73728") == 0);
        assert(strcmp(header(req, "X-Capture-Clock"), "monotonic") == 0);
        assert(strcmp(header(req, "X-Capture-Stage"), "driver-soe") == 0);
        assert(strlen(header(req, "X-Capture-Boot-Id")) == 36);
        uint64_t send_ns = strtoull(header(req, "X-Capture-Send-Monotonic-Ns"), NULL, 10);
        assert(send_ns > 0 && send_ns <= frame_capture_now_ns());
        assert(strstr(header(req, "Access-Control-Expose-Headers"), "X-Capture-Sequence"));
        evhttp_request_free(req);
    }
    out->capture_sequence = 0; out->capture_flags = 0;
    struct evhttp_request *req = snapshot(&server);
    assert(strcmp(header(req, "X-Capture-Sequence"), "0") == 0);
    assert(strcmp(header(req, "X-Capture-Clock"), "unknown") == 0);
    assert(strcmp(header(req, "X-Capture-Stage"), "driver-eof") == 0);
    evhttp_request_free(req);
    out->online = false;
    req = snapshot(&server);
    assert(header(req, "X-Capture-Protocol") == NULL);
    evhttp_request_free(req);
    /* USB/IR frames have no V4L2 metadata; never invent a source timestamp. */
    frame_s *ir = frame_init(); ir->online = true;
    frame_set_data(ir, (const uint8_t *)"JPEG", 4);
    frame_copy(ir, out);
    req = snapshot(&server);
    assert(header(req, "X-Capture-Protocol") == NULL);
    assert(out->capture_timestamp_ns == 0 && out->capture_sequence == 0);
    evhttp_request_free(req);
    frame_destroy(ir); frame_destroy(out); frame_destroy(jpeg); frame_destroy(raw);
    assert(replies == 5);
    puts("PASS: native frame copies, exact HTTP metadata, repeated/zero sequence, unknown clock, offline and IR paths");
    return 0;
}
