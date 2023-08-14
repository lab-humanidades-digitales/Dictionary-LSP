/* eslint-disable no-debugger */
import { useRef, useState } from "react";

export const usePersistentConfig = () => {


    const [storedDeviceId, setStoredDeviceIdInner] = useState(localStorage["search-sign-device-id"]);

    const setStoredDeviceId = (deviceId) => {
        localStorage["search-sign-device-id"] = deviceId;
        setStoredDeviceIdInner(deviceId);
    };

    const [showRecordingDebugInfo] = useState(
        localStorage["search-sign-recording-debug-info"] === "true"
    );

    const [showRecordingPlayStopButtons] = useState(
        localStorage["search-sign-recording-show-play-stop"] === "true"
    );

    const [hideRecordingInstructionsOnLoad, setHideRecordingInstructionsOnLoadInner] = useState(
        localStorage["search-sign-instructions-show-on-load"] === "true"
    );

    const setHideRecordingInstructionsOnLoad = (showOnLoad) => {
        localStorage["search-sign-instructions-show-on-load"] = showOnLoad;
        setHideRecordingInstructionsOnLoadInner(showOnLoad);
    };



    return {
        storedDeviceId,
        setStoredDeviceId,
        showRecordingPlayStopButtons,
        hideRecordingInstructionsOnLoad,
        setHideRecordingInstructionsOnLoad,
        showRecordingDebugInfo
    };
};
