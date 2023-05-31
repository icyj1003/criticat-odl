var ws = new WebSocket('ws://localhost:8001')

ws.onmessage = function (event) {
    console.log(event.data);
};

function removeElementWithAnimation(element, duration = 500) {
    element.style.transition = `opacity ${duration}ms ease-in-out`;
    element.style.opacity = '0';
    setTimeout(() => {
        // element.remove();
    }, duration);
}

function clickButtonOn(article) {
    article.querySelectorAll('span').forEach(function (element) {
        if (element.innerText == "Xem thêm") {
            element.click()
        }
    });
}

function removeElementById(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

function URLdecode(url) {
    return url.replaceAll(" ", "").replaceAll("\\3a", ":").replaceAll("\\26", "&").replaceAll("\\3d", "=").slice(1, -1);
}

function init() {
    // remove story tray
    // removeElementById("MStoriesTray")
    // removeElementById("MComposer")

    // const button = document.createElement("div");
    // button.className = "ex"
    // button.style.position = "fixed"
    // button.style.right = "10px"
    // button.style.bottom = "10px"
    // button.style.width = "50px"
    // button.style.height = "50px"
    // button.style.background = "red"
    // button.style.innerHTML = "Hello"
    // document.body.appendChild(button);

    // handling first batch
    try {
        document.querySelectorAll("article").forEach(article => { handleArcticle(article) })
    } catch (err) { };

}
function articleScraper(article) {
    const body = article.querySelector("div.story_body_container")
    const footer = article.querySelector("footer._22rc")
    const header = body.querySelector("header")
    const content = body.querySelector("div._5rgt._5nk5._5msi")
    const visual = body.querySelector("div._5rgu._7dc9._27x0")
    var imageArray = [];
    if (visual) {
        try {
            visual.querySelectorAll("i").forEach(function (i) {
                imageArray.push(URLdecode(i.getAttribute("style").split("(")[1].split(")")[0]))
            })
        } catch (e) {
        }
    }
    var num_comment_post = 0
    var num_share_post = 0
    footer.querySelectorAll("span._1j-c").forEach(function (e) {
        if (e.innerText.includes("bình luận")) {
            num_comment_post = e.innerText;
        }
        else {
            num_share_post = e.innerText;
        }
    })

    const data = {
        "images": imageArray,
        "post_message": content.innerText,
        "metadata": JSON.parse(article.getAttribute("data-ft")),
        "num_like_post": ((footer.querySelector("div._1g06")) ? footer.querySelector("div._1g06").innerText : 0),
        "num_share_post": num_share_post,
        "num_comment_post": num_comment_post
    }
    return data
}

function handleArcticle(article) {
    clickButtonOn(article);
    if (article.getAttribute("is_handled") != null) { return; }
    article.setAttribute("is_handled", true);

    // change style
    article.style.margin = "20px 20px 0px 20px"
    article.style.padding = "0px 10px 10px 10px"
    article.style.borderRadius = "10px"

    // create label element
    const labelElement = document.createElement("div");
    labelElement.id = "label";
    labelElement.className = "row";
    labelElement.innerHTML = `
    <button class="reliable">Reliable</button>
    <button class="unreliable">Unreliable</button>
    `;

    // Add click event listener to the first button
    const button1 = labelElement.querySelector(".reliable");
    button1.addEventListener("click", () => {
        const data = {
            "data": articleScraper(article),
            "label": 0
        }
        try {
            ws.send(JSON.stringify(data))
            article.style.backgroundColor = "#b3ffb3"
            labelElement.innerHTML = "<h4>✅This article is reliable</h4>";
        }
        catch (err) {
            alert("Websocket not found");
        };


    });

    // Add click event listener to the second button
    const button2 = labelElement.querySelector(".unreliable");
    button2.addEventListener("click", () => {
        const data = {
            "data": articleScraper(article),
            "label": 1
        }
        try {
            ws.send(JSON.stringify(data))
            article.style.backgroundColor = "#ffcccc"
            labelElement.innerHTML = "<h4>❌This article is unreliable</h4>";
        }
        catch (err) {
            alert("Websocket not found");
        };

    });

    article.appendChild(labelElement);
}

var observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
        if (previousUrl != location.href) {
            previousUrl = location.href;
            init();
        }
        if (mutation.type == "childList" && mutation.addedNodes.length > 0) {
            mutation.addedNodes.forEach(function (node) {
                try {
                    node.querySelectorAll("article._55wo._5rgr._5gh8.async_like").forEach(article => { handleArcticle(article) })
                } catch (err) { };
            })
        }
    });
});


// Config info for the observer.
var config = {
    attributes: true,
    childList: true,
    subtree: true,
    characterData: true,
};

let previousUrl = "";
init();
observer.observe(document.body, config);