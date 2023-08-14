let signs = {
    data: [
        {
            name: "red",
            category: "colour",
            cant:1,
            image: "images/colours/red.png",
            url:"videos/red.mp4"
        },
        {
            name: "blue",
            category: "colour",
            cant:1,
            image: "images/colours/blue.png",
            url:"videos/blue.mp4"
        },
        {
            name: "father",
            category: "family",
            cant:1,
            image: "images/family/father.png",
            url:"videos/father.mp4"
        },
        {
            name: "grandmother",
            category: "family",
            cant:2,
            image: "images/family/grandmother.png",
            url:"videos/grandmother.mp4"
        },
        {
            name: "pineapple",
            category: "fruits",
            cant:1,
            image: "images/fruits/pineapple.png",
            url:"videos/pineapple.mp4"
        },
        {
            name: "strawberry",
            category: "fruits",
            cant:1,
            image: "images/fruits/strawberry.png",
            url:"videos/strawberry.mp4"
        },
        {
            name: "close",
            category: "verbs",
            cant:1,
            image: "images/verbs/close.png",
            url:"videos/close.mp4"
        },
        {
            name: "open",
            category: "verbs",
            cant:1,
            image: "images/verbs/open.png",
            url:"videos/open.mp4"
        },
    ],
};

const search = ()=>{
    const searchBox = document.getElementById("inputSearch").value.toLowerCase();
    const productsItem = document.getElementById("products-list");
    const product = document.querySelectorAll(".product-item");
    const sign_name = document.getElementsByTagName("a");

    for(var i=0; i<sign_name.length; i++){
        let match = product[i].getElementsByTagName('a')[0];

        if(match){
            let value_text = match.textContent || match.innerHTML

            if(value_text.toLowerCase().indexOf(searchBox)>-1){
                product[i].style.display=""
            }
            else{
                product[i].style.display="none"
            }

        }
    }

}