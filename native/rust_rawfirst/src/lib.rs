use std::ffi::CStr;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::slice;

#[derive(Clone, Copy, Debug)]
struct Bucket {
    bucket_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

fn latest_complete_bucket_start_ms(complete_through_ms: i64) -> Option<i64> {
    let bucket_ms = 1_000_i64;
    if complete_through_ms < bucket_ms - 1 {
        return None;
    }
    Some((((complete_through_ms + 1) / bucket_ms) * bucket_ms) - bucket_ms)
}

fn crc32(payload: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in payload {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn encode_wal_record(
    ts_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
) -> Option<[u8; 64]> {
    if ts_ms % 1_000 != 0 {
        return None;
    }

    let mut body = Vec::with_capacity(56);
    body.push(1u8);
    body.push(0u8);
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&64u32.to_le_bytes());
    body.extend_from_slice(&ts_ms.to_le_bytes());
    body.extend_from_slice(&open.to_le_bytes());
    body.extend_from_slice(&high.to_le_bytes());
    body.extend_from_slice(&low.to_le_bytes());
    body.extend_from_slice(&close.to_le_bytes());
    body.extend_from_slice(&volume.to_le_bytes());
    let crc = crc32(&body);

    let mut record = [0u8; 64];
    record[0..4].copy_from_slice(b"LQWB");
    record[4..60].copy_from_slice(&body);
    record[60..64].copy_from_slice(&crc.to_le_bytes());
    Some(record)
}

fn aggregate_present_buckets(
    timestamps_ms: &[i64],
    prices: &[f64],
    quantities: &[f64],
    start_filter_ms: Option<i64>,
    end_filter_ms: Option<i64>,
    effective_complete_through_ms: i64,
) -> Vec<Bucket> {
    let mut buckets: Vec<Bucket> = Vec::new();

    for idx in 0..timestamps_ms.len() {
        let ts_ms = timestamps_ms[idx];
        if let Some(start_ms) = start_filter_ms {
            if ts_ms < start_ms {
                continue;
            }
        }
        if let Some(end_ms) = end_filter_ms {
            if ts_ms > end_ms {
                continue;
            }
        }
        if ts_ms > effective_complete_through_ms {
            continue;
        }

        let bucket_ms = (ts_ms / 1_000) * 1_000;
        let price = prices[idx];
        let quantity = quantities[idx];

        match buckets.last_mut() {
            Some(last) if last.bucket_ms == bucket_ms => {
                if price > last.high {
                    last.high = price;
                }
                if price < last.low {
                    last.low = price;
                }
                last.close = price;
                last.volume += quantity;
            }
            _ => buckets.push(Bucket {
                bucket_ms,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: quantity,
            }),
        }
    }

    buckets
}

#[no_mangle]
pub extern "C" fn aggregate_raw_aggtrades_to_1s(
    timestamps_ms: *const i64,
    prices: *const f64,
    quantities: *const f64,
    len: i32,
    range_start_ms: i64,
    has_range_start_ms: i32,
    range_end_ms: i64,
    has_range_end_ms: i32,
    previous_close: f64,
    has_previous_close: i32,
    complete_through_ms: i64,
    out_timestamps_ms: *mut i64,
    out_open: *mut f64,
    out_high: *mut f64,
    out_low: *mut f64,
    out_close: *mut f64,
    out_volume: *mut f64,
    out_capacity: i32,
    out_len: *mut i32,
) -> i32 {
    if timestamps_ms.is_null()
        || prices.is_null()
        || quantities.is_null()
        || out_timestamps_ms.is_null()
        || out_open.is_null()
        || out_high.is_null()
        || out_low.is_null()
        || out_close.is_null()
        || out_volume.is_null()
        || out_len.is_null()
        || len < 0
        || out_capacity < 0
    {
        return 2;
    }

    let len_usize = len as usize;
    if len_usize == 0 {
        unsafe {
            *out_len = 0;
        }
        return 0;
    }

    let timestamps = unsafe { slice::from_raw_parts(timestamps_ms, len_usize) };
    let prices = unsafe { slice::from_raw_parts(prices, len_usize) };
    let quantities = unsafe { slice::from_raw_parts(quantities, len_usize) };

    let last_complete_second = match latest_complete_bucket_start_ms(complete_through_ms) {
        Some(value) => value,
        None => {
            unsafe {
                *out_len = 0;
            }
            return 0;
        }
    };

    let first_trade_second = (timestamps[0] / 1_000) * 1_000;
    let mut start_second = if has_range_start_ms != 0 {
        let aligned = (range_start_ms / 1_000) * 1_000;
        if has_previous_close == 0 {
            aligned.max(first_trade_second)
        } else {
            aligned
        }
    } else {
        first_trade_second
    };
    let mut end_second = last_complete_second;
    if has_range_end_ms != 0 {
        end_second = end_second.min((range_end_ms / 1_000) * 1_000);
    }
    if end_second < start_second {
        unsafe {
            *out_len = 0;
        }
        return 0;
    }

    let buckets = aggregate_present_buckets(
        timestamps,
        prices,
        quantities,
        if has_range_start_ms != 0 {
            Some(range_start_ms)
        } else {
            None
        },
        if has_range_end_ms != 0 {
            Some(range_end_ms)
        } else {
            None
        },
        complete_through_ms,
    );
    if buckets.is_empty() {
        unsafe {
            *out_len = 0;
        }
        return 0;
    }

    if has_previous_close == 0 {
        start_second = start_second.max(buckets[0].bucket_ms);
    }
    if end_second < start_second {
        unsafe {
            *out_len = 0;
        }
        return 0;
    }

    let expected_len = ((end_second - start_second) / 1_000 + 1).max(0) as usize;
    if expected_len > out_capacity as usize {
        return 3;
    }

    let out_ts = unsafe { slice::from_raw_parts_mut(out_timestamps_ms, out_capacity as usize) };
    let out_open = unsafe { slice::from_raw_parts_mut(out_open, out_capacity as usize) };
    let out_high = unsafe { slice::from_raw_parts_mut(out_high, out_capacity as usize) };
    let out_low = unsafe { slice::from_raw_parts_mut(out_low, out_capacity as usize) };
    let out_close = unsafe { slice::from_raw_parts_mut(out_close, out_capacity as usize) };
    let out_volume = unsafe { slice::from_raw_parts_mut(out_volume, out_capacity as usize) };

    let mut idx = 0usize;
    let mut bucket_idx = 0usize;
    let mut carry_close = if has_previous_close != 0 {
        Some(previous_close)
    } else {
        None
    };

    let mut second = start_second;
    while second <= end_second {
        if bucket_idx < buckets.len() && buckets[bucket_idx].bucket_ms == second {
            let bucket = buckets[bucket_idx];
            out_ts[idx] = second;
            out_open[idx] = bucket.open;
            out_high[idx] = bucket.high;
            out_low[idx] = bucket.low;
            out_close[idx] = bucket.close;
            out_volume[idx] = bucket.volume;
            carry_close = Some(bucket.close);
            bucket_idx += 1;
            idx += 1;
        } else if let Some(close_value) = carry_close {
            out_ts[idx] = second;
            out_open[idx] = close_value;
            out_high[idx] = close_value;
            out_low[idx] = close_value;
            out_close[idx] = close_value;
            out_volume[idx] = 0.0;
            idx += 1;
        }
        second += 1_000;
    }

    unsafe {
        *out_len = idx as i32;
    }
    0
}

#[no_mangle]
pub extern "C" fn append_ohlcv_1s_wal(
    wal_path: *const c_char,
    timestamps_ms: *const i64,
    open: *const f64,
    high: *const f64,
    low: *const f64,
    close: *const f64,
    volume: *const f64,
    len: i32,
    fsync_after_write: i32,
    out_written: *mut i32,
) -> i32 {
    if wal_path.is_null()
        || timestamps_ms.is_null()
        || open.is_null()
        || high.is_null()
        || low.is_null()
        || close.is_null()
        || volume.is_null()
        || out_written.is_null()
        || len < 0
    {
        return 2;
    }

    let wal_path_buf = match unsafe { CStr::from_ptr(wal_path) }.to_str() {
        Ok(text) if !text.is_empty() => PathBuf::from(text),
        _ => return 2,
    };

    let len_usize = len as usize;
    if len_usize == 0 {
        unsafe {
            *out_written = 0;
        }
        return 0;
    }

    let timestamps = unsafe { slice::from_raw_parts(timestamps_ms, len_usize) };
    let opens = unsafe { slice::from_raw_parts(open, len_usize) };
    let highs = unsafe { slice::from_raw_parts(high, len_usize) };
    let lows = unsafe { slice::from_raw_parts(low, len_usize) };
    let closes = unsafe { slice::from_raw_parts(close, len_usize) };
    let volumes = unsafe { slice::from_raw_parts(volume, len_usize) };

    let mut encoded = Vec::with_capacity(len_usize * 64);
    for idx in 0..len_usize {
        let record = match encode_wal_record(
            timestamps[idx],
            opens[idx],
            highs[idx],
            lows[idx],
            closes[idx],
            volumes[idx],
        ) {
            Some(value) => value,
            None => return 5,
        };
        encoded.extend_from_slice(&record);
    }

    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&wal_path_buf)
    {
        Ok(handle) => handle,
        Err(_) => return 4,
    };

    if file.write_all(&encoded).is_err() {
        return 4;
    }
    if file.flush().is_err() {
        return 4;
    }
    if fsync_after_write != 0 && file.sync_data().is_err() {
        return 4;
    }

    unsafe {
        *out_written = len;
    }
    0
}
