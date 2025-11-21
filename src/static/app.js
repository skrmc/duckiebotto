const $ = (id) => document.getElementById(id);

const connectionStatus = $('connectionStatus');
const allAutoBtn = $('allAutoBtn');
const speedLimitSlider = $('speedLimit');
const speedLimitValue = $('speedLimitValue');
const colorToleranceSlider = $('colorTolerance');
const colorToleranceValue = $('colorToleranceValue');
const modeToggles = Array.from(document.querySelectorAll('.mode-toggle'));
const colorInputs = {
    yellow: $('colorYellow'),
    white: $('colorWhite'),
};

const debugElements = {
    mode: $('debugMode'),
    laneD: $('debugLaneD'),
    lanePhi: $('debugLanePhi'),
    inLane: $('debugInLane'),
    tof: $('debugTof'),
    light: $('debugLight'),
    obstacle: $('debugObstacle'),
};

const videoStreams = {
    raw: createStreamConfig('rawCanvas', 'rawFps', 'rawFpsValue', '/video/raw/frame'),
    out: createStreamConfig('outCanvas', 'outFps', 'outFpsValue', '/video/out/frame'),
};

function createStreamConfig(canvasId, sliderId, labelId, path) {
    const canvas = $(canvasId);
    const slider = $(sliderId);
    return {
        canvas,
        ctx: canvas ? canvas.getContext('2d') : null,
        slider,
        label: $(labelId),
        path,
        intervalMs: slider ? 1000 / Number(slider.value || 10) : 100,
        running: false,
    };
}

const manualKeys = new Set(['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);
const pressedKeys = new Set();
let manualLoop = null;
let socket = null;
let reconnectTimer = null;
let currentState = {};
let speedSliderActive = false;
let toleranceSliderActive = false;
const COLOR_HEX_REGEX = /^#[0-9A-F]{6}$/i;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setConnectionStatus(title, colorClass) {
    if (!connectionStatus) return;
    connectionStatus.title = title;
    connectionStatus.className = `w-4 h-4 rounded-full ${colorClass}`;
}

function normalizeHexValue(value) {
    if (typeof value !== 'string') return null;
    let trimmed = value.trim();
    if (!trimmed) return null;
    if (!trimmed.startsWith('#')) trimmed = `#${trimmed}`;
    return COLOR_HEX_REGEX.test(trimmed) ? trimmed.toUpperCase() : null;
}

function setColorInputValue(input, value) {
    input.value = value;
    input.dataset.current = value;
    delete input.dataset.pending;
}

function handleColorInput(colorKey, input) {
    const normalized = normalizeHexValue(input.value);
    if (!normalized) {
        input.setCustomValidity('Use HEX format like #FFD400');
        input.reportValidity();
        return;
    }
    input.setCustomValidity('');
    const previous = input.dataset.current;
    setColorInputValue(input, normalized);
    if (previous === normalized) return;
    sendMessage({ type: 'set_color_reference', color: colorKey, value: normalized });
}

function bindColorInputs() {
    Object.entries(colorInputs).forEach(([key, input]) => {
        if (!input) return;
        input.addEventListener('change', () => handleColorInput(key, input));
        input.addEventListener('blur', () => {
            if (input.dataset.pending) {
                setColorInputValue(input, input.dataset.pending);
            }
        });
    });
}

function updateColorInputs(refs) {
    Object.entries(colorInputs).forEach(([key, input]) => {
        if (!input) return;
        const raw = typeof refs[key] === 'string' ? refs[key] : null;
        const normalized = normalizeHexValue(raw);
        if (!normalized) return;
        if (document.activeElement === input) {
            input.dataset.pending = normalized;
            return;
        }
        setColorInputValue(input, normalized);
    });
}

function connectSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${window.location.host}/ws`);

    socket.addEventListener('open', () => {
        setConnectionStatus('Connected', 'bg-emerald-500');
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    });

    socket.addEventListener('message', (event) => {
        try {
            const payload = JSON.parse(event.data);
            if (payload.type === 'state') {
                applyState(payload.data);
            } else if (payload.type === 'error') {
                setConnectionStatus(payload.message || 'Error', 'bg-rose-500');
            }
        } catch (error) {
            console.error('Invalid WS message', error);
        }
    });

    socket.addEventListener('close', () => {
        setConnectionStatus('Disconnected - retrying...', 'bg-amber-500');
        stopManualLoop();
        pressedKeys.clear();
        if (!reconnectTimer) {
            reconnectTimer = setTimeout(connectSocket, 1500);
        }
    });

    socket.addEventListener('error', () => {
        setConnectionStatus('Connection error', 'bg-rose-500');
    });
}

function sendMessage(message) {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        return false;
    }
    socket.send(JSON.stringify(message));
    return true;
}

modeToggles.forEach((input) => {
    input.addEventListener('change', (event) => {
        const mode = input.dataset.mode;
        if (!mode) return;
        sendMessage({ type: 'set_mode', mode, enabled: event.target.checked });
    });
});

if (allAutoBtn) {
    allAutoBtn.addEventListener('click', () => {
        const isAllAuto = Boolean(
            currentState.lane_follow_enabled && currentState.obstacle_avoid_enabled && currentState.traffic_light_enabled
        );
        const target = isAllAuto ? 'manual' : 'all_auto';
        sendMessage({ type: 'set_mode', mode: target, enabled: true });
    });
}

function prettyNumber(value) {
    return typeof value === 'number' && !Number.isNaN(value) ? value.toFixed(2) : '--';
}

function updateAllAutoButton(state) {
    if (!allAutoBtn) return;
    const isAllAuto = state.lane_follow_enabled && state.obstacle_avoid_enabled && state.traffic_light_enabled;
    allAutoBtn.textContent = isAllAuto ? 'Switch to Manual Drive' : 'Enable All Auto Functions';
    allAutoBtn.classList.toggle('bg-emerald-500', isAllAuto);
    allAutoBtn.classList.toggle('hover:bg-emerald-600', isAllAuto);
    allAutoBtn.classList.toggle('bg-slate-500', !isAllAuto);
    allAutoBtn.classList.toggle('hover:bg-slate-600', !isAllAuto);
}

function applyState(state) {
    currentState = state;
    modeToggles.forEach((input) => {
        const key = input.dataset.state;
        if (key && key in state) {
            input.checked = Boolean(state[key]);
        }
    });
    updateAllAutoButton(state);

    if (!state.manual_mode) {
        pressedKeys.clear();
        stopManualLoop();
    }

    debugElements.mode && (debugElements.mode.textContent = state.mode);
    debugElements.laneD && (debugElements.laneD.textContent = prettyNumber(state.lane_error));
    debugElements.lanePhi && (debugElements.lanePhi.textContent = prettyNumber(state.heading_error));
    if (debugElements.inLane) {
        let value = '--';
        if (typeof state.lane_in_lane === 'boolean') {
            value = state.lane_in_lane ? 'yes' : 'no';
        }
        debugElements.inLane.textContent = value;
    }
    debugElements.tof && (debugElements.tof.textContent = typeof state.tof_distance === 'number' ? `${prettyNumber(state.tof_distance)} m` : '--');
    debugElements.light && (debugElements.light.textContent = state.traffic_light);
    debugElements.obstacle && (debugElements.obstacle.textContent = state.obstacle_detected ? 'yes' : 'no');

    if (speedLimitSlider && typeof state.speed_limit === 'number' && !speedSliderActive) {
        const sliderValue = ratioToSliderValue(state.speed_limit);
        speedLimitSlider.value = String(sliderValue);
        updateSpeedLimitLabel(sliderValue);
    } else if (speedSliderActive && speedLimitSlider) {
        updateSpeedLimitLabel(Number(speedLimitSlider.value));
    }

    if (colorToleranceSlider && typeof state.color_tolerance_scale === 'number' && !toleranceSliderActive) {
        const sliderValue = toleranceScaleToSliderValue(state.color_tolerance_scale);
        colorToleranceSlider.value = String(sliderValue);
        updateColorToleranceLabel(sliderValue);
    } else if (toleranceSliderActive && colorToleranceSlider) {
        updateColorToleranceLabel(Number(colorToleranceSlider.value));
    }

    if (state.color_refs && typeof state.color_refs === 'object') {
        updateColorInputs(state.color_refs);
    }
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function ratioToSliderValue(ratio) {
    return Math.round(clamp(ratio, 0.2, 1.0) * 100);
}

function sliderValueToRatio(value) {
    return clamp(Number(value) / 100, 0.2, 1.0);
}

function toleranceScaleToSliderValue(scale) {
    return Math.round(clamp(scale, 0.5, 2.0) * 100);
}

function toleranceSliderValueToScale(value) {
    return clamp(Number(value) / 100, 0.5, 2.0);
}

function updateSpeedLimitLabel(value) {
    if (!speedLimitValue) return;
    const numeric = Number(value);
    speedLimitValue.textContent = numeric >= 100 ? 'No limit' : `${numeric}%`;
}

function updateColorToleranceLabel(value) {
    if (!colorToleranceValue) return;
    const numeric = Number(value);
    colorToleranceValue.textContent = `${numeric}%`;
}

function bindSlider(slider, { onPreview, onCommit }) {
    if (!slider) return;
    let active = false;
    slider.addEventListener('input', (event) => {
        active = true;
        const value = Number(event.target.value);
        onPreview && onPreview(value);
    });
    const finish = () => {
        if (!active) return;
        active = false;
        onCommit && onCommit(Number(slider.value));
    };
    slider.addEventListener('change', finish);
    slider.addEventListener('mouseup', finish);
    slider.addEventListener('touchend', finish);
    slider.addEventListener('touchcancel', finish);
}

Object.values(videoStreams).forEach((stream) => {
    if (!stream.slider) return;
    const initial = Number(stream.slider.value || 10);
    setStreamFps(stream, initial);
    bindSlider(stream.slider, {
        onPreview: (value) => {
            if (stream.label) stream.label.textContent = `${value} FPS`;
        },
        onCommit: (value) => setStreamFps(stream, value),
    });
});

function setStreamFps(stream, value) {
    stream.intervalMs = 1000 / value;
    if (stream.label) {
        stream.label.textContent = `${value} FPS`;
    }
}

if (speedLimitSlider) {
    updateSpeedLimitLabel(Number(speedLimitSlider.value));
    bindSlider(speedLimitSlider, {
        onPreview: (value) => {
            speedSliderActive = true;
            updateSpeedLimitLabel(value);
        },
        onCommit: (value) => {
            if (!speedSliderActive) return;
            speedSliderActive = false;
            updateSpeedLimitLabel(value);
            sendMessage({ type: 'set_speed_limit', value: sliderValueToRatio(value) });
        },
    });
}

if (colorToleranceSlider) {
    updateColorToleranceLabel(Number(colorToleranceSlider.value));
    bindSlider(colorToleranceSlider, {
        onPreview: (value) => {
            toleranceSliderActive = true;
            updateColorToleranceLabel(value);
        },
        onCommit: (value) => {
            if (!toleranceSliderActive) return;
            toleranceSliderActive = false;
            updateColorToleranceLabel(value);
            sendMessage({ type: 'set_color_tolerance', value: toleranceSliderValueToScale(value) });
        },
    });
}

function computeManualCommand() {
    let linear = 0;
    if (pressedKeys.has('w') || pressedKeys.has('arrowup')) linear += 0.3;
    if (pressedKeys.has('s') || pressedKeys.has('arrowdown')) linear -= 0.3;

    let angular = 0;
    if (pressedKeys.has('a') || pressedKeys.has('arrowleft')) angular += 0.5;
    if (pressedKeys.has('d') || pressedKeys.has('arrowright')) angular -= 0.5;

    linear = clamp(linear, -0.4, 0.4);
    angular = clamp(angular, -0.6, 0.6);
    const left = clamp(linear - angular, -1.0, 1.0);
    const right = clamp(linear + angular, -1.0, 1.0);
    return { left, right };
}

function startManualLoop() {
    if (manualLoop) return;
    manualLoop = setInterval(() => {
        if (!currentState.manual_mode || pressedKeys.size === 0) {
            stopManualLoop();
            return;
        }
        const command = computeManualCommand();
        sendMessage({ type: 'manual', left: command.left, right: command.right });
    }, 100);
}

function stopManualLoop() {
    if (manualLoop) {
        clearInterval(manualLoop);
        manualLoop = null;
    }
}

function publishManual() {
    if (!currentState.manual_mode) {
        stopManualLoop();
        return;
    }
    const command = computeManualCommand();
    sendMessage({ type: 'manual', left: command.left, right: command.right });
    pressedKeys.size > 0 ? startManualLoop() : stopManualLoop();
}

document.addEventListener('keydown', (event) => {
    const key = event.key.toLowerCase();
    if (!manualKeys.has(key) || pressedKeys.has(key) || !currentState.manual_mode) return;
    event.preventDefault();
    pressedKeys.add(key);
    publishManual();
});

document.addEventListener('keyup', (event) => {
    const key = event.key.toLowerCase();
    if (!manualKeys.has(key) || !currentState.manual_mode) return;
    event.preventDefault();
    pressedKeys.delete(key);
    publishManual();
});

async function drawBlob(canvas, ctx, blob) {
    if (!ctx || !blob || blob.size === 0) return;
    if ('createImageBitmap' in window) {
        const bitmap = await createImageBitmap(blob);
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close();
        return;
    }
    await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(img.src);
            resolve();
        };
        img.onerror = (error) => {
            URL.revokeObjectURL(img.src);
            reject(error);
        };
        img.src = URL.createObjectURL(blob);
    });
}

function startStreamLoop(stream) {
    if (!stream.canvas || !stream.ctx || stream.running) return;
    stream.running = true;
    (async function loop() {
        while (stream.running) {
            const start = performance.now();
            try {
                const response = await fetch(`${stream.path}?ts=${Date.now()}`, { cache: 'no-store' });
                if (response.status === 200) {
                    const blob = await response.blob();
                    await drawBlob(stream.canvas, stream.ctx, blob);
                }
            } catch (error) {
                console.warn('Stream error', stream.path, error);
                await sleep(800);
            }
            const elapsed = performance.now() - start;
            const delay = Math.max(0, stream.intervalMs - elapsed);
            await sleep(delay);
        }
    })();
}

Object.values(videoStreams).forEach(startStreamLoop);
bindColorInputs();
connectSocket();
