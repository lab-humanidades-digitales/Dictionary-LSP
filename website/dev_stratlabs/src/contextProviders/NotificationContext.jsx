import PropTypes from "prop-types";
import { useState, useEffect, useMemo, createContext, ReactNode, useContext } from "react";
import { createPortal } from "react-dom";

import NotificationUI from "components/Notification/Notification";
import { AxiosError } from "axios";

const generateUUID = () => {
  let first = (Math.random() * 46656) | 0;
  let second = (Math.random() * 46656) | 0;
  first = ("000" + first.toString(36)).slice(-3);
  second = ("000" + second.toString(36)).slice(-3);

  return first + second;
};

const contextDefaultValue = {
  open: () => {},
  throwError: () => {},
};

const NotificationContext = createContext(contextDefaultValue);
export const useNotification = () => useContext(NotificationContext);

export const NotificationProvider = ({ children }) => {
  const [isBrowser, setIsBrowser] = useState(false);
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    setIsBrowser(true);
  }, []);

  const open = (params) => {
    setNotifications((currentNotifications) => [
      ...currentNotifications,
      { id: generateUUID(), ...params },
    ]);
  };

  const close = (id) =>
    setNotifications((currentNotifications) =>
      currentNotifications.filter((notification) => notification.id !== id)
    );

  const throwError = (ex) => {
    if (!ex.response?.data?.error) throw ex;

    const errorData = ex.response.data.error;

    if (errorData.type === "AUTH")
      open({
        title: "Permisos insuficientes",
        subTitle: errorData.message || "Ha ocurrido un error.",
        type: "error",
      });
    else if (errorData.type === "VAL")
      open({
        title: "Error de validación",
        subTitle: errorData.message || "Ha ocurrido un error.",
        type: "warning",
      });
    else if (errorData.type === "EXC")
      open({
        title: "Error",
        subTitle: errorData.message || "Ha ocurrido un error.",
        type: "error",
      });
    else
      open({
        title: "Error",
        subTitle: errorData.message || "Ha ocurrido un error.",
        type: "error",
      });
  };

  const contextValue = useMemo(() => ({ open, throwError }), []);

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}

      {isBrowser
        ? createPortal(
            <div className="fixed right-0 z-20 flex flex-col gap-2 top-24">
              {notifications.map((notification) => (
                <NotificationUI
                  {...notification}
                  key={notification.id}
                  onClose={() => close(notification?.id)}
                />
              ))}
            </div>,
            document.getElementById("notification-root")
          )
        : null}
    </NotificationContext.Provider>
  );
};

NotificationProvider.propTypes = {
  children: PropTypes.node,
};
