// Dean Attali / Beautiful Jekyll 2016

var main = {

  bigImgEl : null,
  numImgs : null,

  init : function() {
    // Shorten the navbar after scrolling a little bit down
    $(window).scroll(function() {
        if ($(".navbar").offset().top > 50) {
            $(".navbar").addClass("top-nav-short");
            $(".navbar-custom .avatar-container").fadeOut(500);
        } else {
            $(".navbar").removeClass("top-nav-short");
            $(".navbar-custom .avatar-container").fadeIn(500);
        }

        // Reading progress bar (post pages only)
        if ($('.blog-post').length > 0) {
          var winScroll = document.documentElement.scrollTop || document.body.scrollTop;
          var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
          var scrolled = height > 0 ? (winScroll / height) * 100 : 0;
          $('#reading-progress-bar').css('width', scrolled + '%');
        }

        // Back to top button visibility
        if ($(window).scrollTop() > 300) {
          $('#back-to-top').addClass('visible');
        } else {
          $('#back-to-top').removeClass('visible');
        }
    });

    // On mobile, hide the avatar when expanding the navbar menu
    $('#main-navbar').on('show.bs.collapse', function () {
      $(".navbar").addClass("top-nav-expanded");
    });
    $('#main-navbar').on('hidden.bs.collapse', function () {
      $(".navbar").removeClass("top-nav-expanded");
    });

    // On mobile, when clicking on a multi-level navbar menu, show the child links
    $('#main-navbar').on("click", ".navlinks-parent", function(e) {
      var target = e.target;
      $.each($(".navlinks-parent"), function(key, value) {
        if (value == target) {
          $(value).parent().toggleClass("show-children");
        } else {
          $(value).parent().removeClass("show-children");
        }
      });
    });

    // Ensure nested navbar menus are not longer than the menu header
    var menus = $(".navlinks-container");
    if (menus.length > 0) {
      var navbar = $("#main-navbar ul");
      var fakeMenuHtml = "<li class='fake-menu' style='display:none;'><a></a></li>";
      navbar.append(fakeMenuHtml);
      var fakeMenu = $(".fake-menu");

      $.each(menus, function(i) {
        var parent = $(menus[i]).find(".navlinks-parent");
        var children = $(menus[i]).find(".navlinks-children a");
        var words = [];
        $.each(children, function(idx, el) { words = words.concat($(el).text().trim().split(/\s+/)); });
        var maxwidth = 0;
        $.each(words, function(id, word) {
          fakeMenu.html("<a>" + word + "</a>");
          var width =  fakeMenu.width();
          if (width > maxwidth) {
            maxwidth = width;
          }
        });
        $(menus[i]).css('min-width', maxwidth + 'px')
      });

      fakeMenu.remove();
    }

    // Code block language badges
    $('pre code[class*="language-"]').each(function() {
      var match = $(this).attr('class').match(/language-(\w+)/);
      if (match) {
        $(this).closest('pre').attr('data-lang', match[1]);
      }
    });

    // Show reading progress bar on post pages
    if ($('.blog-post').length > 0) {
      $('#reading-progress-bar').css('display', 'block');
    }

    // Back to top click handler
    $('#back-to-top').on('click', function() {
      $('html, body').animate({ scrollTop: 0 }, 400);
    });

    // show the big header image
    main.initImgs();
    main.initTOC();
  },

  initImgs : function() {
    // If the page was large images to randomly select from, choose an image
    if ($("#header-big-imgs").length > 0) {
      main.bigImgEl = $("#header-big-imgs");
      main.numImgs = main.bigImgEl.attr("data-num-img");

          // 2fc73a3a967e97599c9763d05e564189
	  // set an initial image
	  var imgInfo = main.getImgInfo();
	  var src = imgInfo.src;
	  var desc = imgInfo.desc;
  	  main.setImg(src, desc);

	  // For better UX, prefetch the next image so that it will already be loaded when we want to show it
  	  var getNextImg = function() {
	    var imgInfo = main.getImgInfo();
	    var src = imgInfo.src;
	    var desc = imgInfo.desc;

		var prefetchImg = new Image();
  		prefetchImg.src = src;
		// if I want to do something once the image is ready: `prefetchImg.onload = function(){}`

  		setTimeout(function(){
                  var img = $("<div></div>").addClass("big-img-transition").css("background-image", 'url(' + src + ')');
  		  $(".intro-header.big-img").prepend(img);
  		  setTimeout(function(){ img.css("opacity", "1"); }, 50);

		  // after the animation of fading in the new image is done, prefetch the next one
  		  //img.one("transitioned webkitTransitionEnd oTransitionEnd MSTransitionEnd", function(){
		  setTimeout(function() {
		    main.setImg(src, desc);
			img.remove();
  			getNextImg();
		  }, 1000);
  		  //});
  		}, 6000);
  	  };

	  // If there are multiple images, cycle through them
	  if (main.numImgs > 1) {
  	    getNextImg();
	  }
    }
  },

  getImgInfo : function() {
  	var randNum = Math.floor((Math.random() * main.numImgs) + 1);
    var src = main.bigImgEl.attr("data-img-src-" + randNum);
	var desc = main.bigImgEl.attr("data-img-desc-" + randNum);

	return {
	  src : src,
	  desc : desc
	}
  },

  setImg : function(src, desc) {
	$(".intro-header.big-img").css("background-image", 'url(' + src + ')');
	if (typeof desc !== typeof undefined && desc !== false) {
	  $(".img-desc").text(desc).show();
	} else {
	  $(".img-desc").hide();
	}
  },

  initTOC: function() {
    if ($('.blog-post').length === 0) return;
    var headings = $('.blog-post').find('h2, h3');
    if (headings.length < 2) return;

    var nav = $('<nav id="toc"><div class="toc-label">Contents</div><ul class="toc-list"></ul></nav>');
    var list = nav.find('.toc-list');

    headings.each(function() {
      var $h = $(this);
      var tag = this.tagName.toLowerCase();
      var text = $h.text();
      if (!$h.attr('id')) {
        var id = text.toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-')
          .replace(/-+/g, '-')
          .trim();
        $h.attr('id', id);
      }
      var li = $('<li><a href="#' + $h.attr('id') + '">' + text + '</a></li>');
      if (tag === 'h3') li.addClass('toc-l2');
      list.append(li);
    });

    $('body').append(nav);

    if (!('IntersectionObserver' in window)) return;
    var tocLinks = nav.find('a');
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var id = '#' + entry.target.getAttribute('id');
          tocLinks.removeClass('toc-active');
          nav.find('a[href="' + id + '"]').addClass('toc-active');
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });

    headings.each(function() {
      if (this.id) observer.observe(this);
    });
  }
};

// 2fc73a3a967e97599c9763d05e564189

document.addEventListener('DOMContentLoaded', main.init);
