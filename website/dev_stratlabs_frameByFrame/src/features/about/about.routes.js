import DefaultLayout from "layouts/DefaultLayout"
import Team from "./views/Team";

// eslint-disable-next-line no-unused-vars
const routes = [
    {
        route: "/team",
        component: (
            <DefaultLayout>
                <Team></Team>
            </DefaultLayout>
        ),
    },
];

export default routes;
