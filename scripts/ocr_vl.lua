-- Persistent local VL OCR processes, one per profile (see ocr_vl.py).
local M = {}
local python = os.getenv("HOME") .. "/.hammerspoon/.venv-vl/bin/python"
local script = os.getenv("HOME") .. "/.hammerspoon/scripts/ocr_vl.py"

-- idleSeconds: unload the model after this long without requests (nil keeps it loaded).
function M.new(profile, idleSeconds)
    local W = {profile = profile}
    local task, pending, buffer, errors, idleTimer = nil, nil, "", "", nil
    local nextId = 0

    local function finish(code, text, err)
        if not pending then return end
        local request = pending
        pending = nil
        if request.timer then request.timer:stop() end
        if idleSeconds and task then
            idleTimer = hs.timer.doAfter(idleSeconds, function()
                if not pending then
                    print("VL worker idle, unloading:", profile)
                    W.stop()
                end
            end)
        end
        request.callback(code, text or "", err or "")
    end

    function W.stop()
        if idleTimer then idleTimer:stop(); idleTimer = nil end
        local old = task
        task = nil
        if old and old:isRunning() then old:terminate() end
        finish(2, "", "VL worker stopped")
    end

    function W.start()
        if task then return true end
        buffer, errors = "", ""
        local newTask
        newTask = hs.task.new(python,
            function(code, _, err)
                if task ~= newTask then return end
                task = nil
                finish(2, "", "VL worker exited (" .. tostring(code) .. "): " .. errors .. (err or ""))
            end,
            function(_, out, err)
                if task ~= newTask then return false end
                errors = (errors .. (err or "")):sub(-4000)
                buffer = buffer .. (out or "")
                while true do
                    local newline = buffer:find("\n", 1, true)
                    if not newline then break end
                    local line = buffer:sub(1, newline - 1)
                    buffer = buffer:sub(newline + 1)
                    local ok, response = pcall(hs.json.decode, line)
                    if ok and type(response) == "table" then
                        if response.ready then
                            print("VL worker ready:", response.backend)
                        elseif pending and response.id == pending.id then
                            if response.error then finish(2, "", response.error)
                            else
                                print("OCR backend:", response.backend, "seconds:", response.seconds)
                                finish(0, response.text)
                            end
                        end
                    elseif line ~= "" then print("VL diagnostic:", line) end
                end
                return true
            end,
            {script, "--worker", "--profile", profile})
        task = newTask
        if not task or not task:start() then
            task = nil
            return false
        end
        return true
    end

    function W.run(image, scale, callback)
        if pending then callback(2, "", "VL OCR is already running"); return end
        if idleTimer then idleTimer:stop(); idleTimer = nil end
        nextId = nextId + 1
        pending = {id = nextId, callback = callback}
        pending.timer = hs.timer.doAfter(180, function()
            local old = task
            task = nil
            if old and old:isRunning() then old:terminate() end
            finish(2, "", "VL OCR timed out; select a smaller region")
        end)
        if not W.start() then
            finish(2, "", "Cannot start VL OCR; check .venv-vl and ocr_vl.py")
            return
        end
        local ok, err = pcall(function()
            task:setInput(hs.json.encode({id = nextId, image = image, scale = scale}) .. "\n")
        end)
        if not ok then
            local old = task
            task = nil
            if old and old:isRunning() then old:terminate() end
            finish(2, "", tostring(err))
        end
    end

    return W
end

return M
