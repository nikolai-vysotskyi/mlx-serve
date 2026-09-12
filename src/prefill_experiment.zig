//! Opt-in same-process research screening; never changes request contents.
const std = @import("std");
const log = @import("log.zig");
threadlocal var cursor: usize = 0;
threadlocal var arm: ?u8 = null;

pub fn begin(tokens: usize) void {
    arm = null;
    if (tokens < 2048) return;
    const raw = std.c.getenv("QWEN4_PREFILL_ARM_SEQUENCE") orelse return;
    const sequence = std.mem.sliceTo(raw, 0);
    if (cursor >= sequence.len) return;
    for (sequence) |c| if (c < '0' or c > '3') return;
    arm = sequence[cursor] - '0';
    log.info("[prefill-experiment] request={d} arm={d} group={} ple={} same_process=true\n", .{ cursor, arm.?, arm.? & 1 != 0, arm.? & 2 != 0 });
    cursor += 1;
}

pub fn group() ?bool {
    return if (arm) |v| v & 1 != 0 else null;
}
pub fn ple() ?bool {
    return if (arm) |v| v & 2 != 0 else null;
}
