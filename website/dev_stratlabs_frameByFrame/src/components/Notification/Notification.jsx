import PropTypes from "prop-types";
import { useTimeOut } from "hooks/useTimeOut";

const Notification = ({ type = "info", title, subTitle, duration = 5000, onClose }) => {
  useTimeOut(onClose, duration);

  return (
    <div
      className={`notification-${type} relative z-40 flex w-90 gap-2 border-l-3 py-6 px-4 shadow-04 ${
        !subTitle && "items-center"
      }`}
    >
      {/*<Icon name={`gl_${type}`} className={`icon-color-inherit icon-${type}`} />*/}
      <div className="flex flex-col">
        <p className={`w-full text-paragraph-06 font-bold text-${type}`}>{title}</p>
        {subTitle && <p className={`w-full text-paragraph-07 text-${type}`}>{subTitle}</p>}
      </div>
    </div>
  );
};

Notification.propTypes = {
  type: PropTypes.oneOf([
    "white",
    "primary",
    "secondary",
    "info",
    "success",
    "warning",
    "error",
    "light",
    "dark",
  ]),
  title: PropTypes.node.isRequired,
  subTitle: PropTypes.string,
  duration: PropTypes.number,
  onClose: PropTypes.func,
};

export default Notification;
