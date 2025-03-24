window.onload = function () {
    const textarea = document.getElementById("send_query")
    textarea.addEventListener("input", function () {
        textarea.style.height = 'auto'
        textarea.style.height = (textarea.scrollHeight > 200 ? 200 : textarea.scrollHeight) + 'px'
    })

    const sidebarOpenBtn = document.getElementById("sidebarbtnopen")
    const sidebarCloseBtn = document.getElementById("sidebarbtnclose")
    const sidebar = document.getElementById("sidebar")
    const sidebarBtnMobile = document.getElementById("sidebarbtnmobile")
    const sidebarCloseBtnMobile = document.getElementById("sidebarbtnclosemobile")
    sidebarOpenBtn.addEventListener("click", function () {
        sidebar.style.width = '260px'
        sidebar.classList.add('p-2')
        sidebarOpenBtn.classList.add('d-md-none')
        setTimeout(function () {
            sidebar.style.minWidth = '260px'
        }, 200)
    })
    sidebarCloseBtn.addEventListener("click", function () {
        sidebar.style.minWidth = '0'
        sidebar.style.width = '0'
        sidebar.classList.remove('p-2')
        sidebarOpenBtn.classList.remove('d-md-none')
        sidebarOpenBtn.classList.add('d-md-block')
    })

    sidebarBtnMobile.addEventListener("click", function () {
        sidebar.style.display = 'block'
        sidebar.style.width = '260px'
        sidebar.style.minWidth = '260px'
        sidebar.classList.add('p-2')
        sidebarCloseBtnMobile.classList.remove('d-none')
    })

    sidebarCloseBtnMobile.addEventListener("click", function () {
        sidebar.style.display = 'none'
        sidebar.style.width = '0'
        sidebar.style.minWidth = '0'
        sidebar.classList.remove('p-2')
        sidebarCloseBtnMobile.classList.add('d-none')
    })

    window.onresize = function () {
        if (window.innerWidth <= 768) {
            sidebar.style.display = 'none'
            sidebar.style.position = 'absolute'
            sidebar.style.zIndex = 9999999
            sidebar.style.minWidth = '0'
            sidebar.style.width = '0'
            sidebar.classList.remove('p-2')
            sidebarCloseBtn.style.display = 'none'
        }
        else {
            sidebar.style.display = 'block'
            sidebar.style.position = 'static'
            sidebar.style.minWidth = '260px'
            sidebar.style.width = '260px'
            sidebar.style.zIndex = 0
            sidebar.classList.add('p-2')
            sidebarCloseBtn.style.display = 'block'
        }
    }

    if (window.innerWidth <= 768) {
        sidebar.style.display = 'none'
        sidebar.style.position = 'absolute'
        sidebar.style.zIndex = 999999
        sidebar.style.minWidth = '0'
        sidebar.style.width = '0'
        sidebar.classList.remove('p-2')
        sidebarCloseBtn.style.display = 'none'
    }
    document.getElementById('tryNowBtn').addEventListener('click', function () {
        let inputField = document.getElementById('send_query');
    
        // Apply styles
        inputField.style.border = "10px solidrgb(255, 255, 255) 255, 255)";
        inputField.style.boxShadow = "0 0 10px rgba(255, 255, 255, 0.5)";
      
        // Ensure reflow for smooth transition
        inputField.offsetHeight;
    
        // Focus on input field
        inputField.focus();
    
        // Remove border and shadow after 1.5 seconds
        setTimeout(() => {
            inputField.style.border = "";
            inputField.style.boxShadow = "";
        }, 2000);
    });
    document.getElementById('sidebarbtnmobile').addEventListener('click', function() {
        document.getElementById('sidebar').classList.toggle('active');
      });
      
      document.getElementById('sidebarbtnclosemobile').addEventListener('click', function() {
        document.getElementById('sidebar').classList.remove('active');
      });
      // Get all the buttons
    const buttons = document.querySelectorAll('.model-btn');

    // Add click event listener to each button
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove 'selected' class from all buttons
            buttons.forEach(btn => btn.classList.remove('selected'));
            // Add 'selected' class to the clicked button
            this.classList.add('selected');
        });
    });
    
}