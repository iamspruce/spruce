"use client";
import { config } from "config";
import Link from "next/link";
import React from "react";
import { Menu, MenuList, MenuButton, MenuLink } from "@reach/menu-button";
import "@reach/menu-button/styles.css";

function MenuBtn() {
  const toggleMenu = (e: any) => {
    document.querySelector(".menu-toggle")?.classList.toggle("active");
    document.querySelector(".header_menu")?.classList.toggle("active");
  };

  return (
    <>
      <Menu>
        <MenuButton aria-label="Navigation button">
          <svg
            className="menu-icon"
            aria-hidden="true"
            width="22"
            height="22"
            xmlns="http://www.w3.org/2000/svg"
            fillRule="evenodd"
            clip-rule="evenodd"
          >
            <path
              d="m22 15.25c0-.414-.336-.75-.75-.75h-18.5c-.414 0-.75.336-.75.75s.336.75.75.75h18.5c.414 0 .75-.336.75-.75zm0-6.5c0-.414-.336-.75-.75-.75h-18.5c-.414 0-.75.336-.75.75s.336.75.75.75h18.5c.414 0 .75-.336.75-.75z"
              fillRule="nonzero"
            />
          </svg>
        </MenuButton>
        <MenuList>
          {config.links.map((link, index) => (
            <li key={link.name} className={`header_menu_list `}>
              <Link
                className={`header_menu_link animate fade delay-${index}`}
                href={link.url}
              >
                {link.name}
              </Link>
            </li>
          ))}
        </MenuList>
      </Menu>
    </>
  );
}
export default MenuBtn;
