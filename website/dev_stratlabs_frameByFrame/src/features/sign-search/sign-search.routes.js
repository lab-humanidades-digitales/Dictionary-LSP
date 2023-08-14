import DefaultLayout from "layouts/DefaultLayout"
import SignSearch from "./views/SignSearch";

const routes = [
    {
        route: "/search-by-sign",
        component: (
            <DefaultLayout>
                <SignSearch></SignSearch>
            </DefaultLayout>
        ),
    },
];

export default routes;
