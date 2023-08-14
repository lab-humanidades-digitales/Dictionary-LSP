import DefaultLayout from "layouts/DefaultLayout"
import TextSearch from "./views/TextSearch";

// eslint-disable-next-line no-unused-vars
const routes = [
    {
        route: "/search-by-text",
        component: (
            <DefaultLayout>
                <TextSearch></TextSearch>
            </DefaultLayout>
        ),
    },
];

export default routes;
