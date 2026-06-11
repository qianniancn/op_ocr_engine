#pragma once

#include "ocr_engine.h"

#include <string>
#include <memory>
#include <functional>

namespace ocr {

class OcrServer {
public:
    using RecognizeFn = std::function<OcrResponse(const uint8_t*, int, int, int)>;
    using VersionFn = std::function<const char*()>;

    /// Construct the server with OCR callbacks.
    /// The supplied callbacks must remain valid for the server lifetime.
    OcrServer(RecognizeFn recognize, VersionFn version);
    ~OcrServer();

    // Non-copyable
    OcrServer(const OcrServer&) = delete;
    OcrServer& operator=(const OcrServer&) = delete;

    /// Start listening on the given host and port (blocking).
    /// @return true if server started successfully
    bool listen(const std::string& host, int port);

    /// Stop the server gracefully.
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ocr
