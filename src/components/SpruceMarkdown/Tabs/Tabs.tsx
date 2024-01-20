import {
  Tabs as ReachTabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
} from "@reach/tabs";
import "@reach/tabs/styles.css";
import React from "react";
import TabImage from "../TabImage";
import Image1 from "/public/img/features/1.png";
import Image2 from "/public/img/features/2.png";
import Image3 from "/public/img/features/3.png";
import Image4 from "/public/img/features/4.png";
import Image5 from "/public/img/features/5.png";
import Image6 from "/public/img/features/6.png";
import Image7 from "/public/img/features/7.png";
import Image8 from "/public/img/features/8.png";
import Image9 from "/public/img/features/9.png";
import Image10 from "/public/img/features/10.png";

function Tabs() {
  const [tabIndex, setTabIndex] = React.useState(0);

  const updateState = () => {
    if (tabIndex < 9) {
      setTabIndex((prev) => prev + 1);
    } else {
      setTabIndex(0);
    }
  };

  React.useEffect(() => {
    const intervalId = setInterval(() => {
      updateState();
    }, 10000);

    return () => clearInterval(intervalId);
  }, [tabIndex]);

  const handleTabsChange = (index: number) => {
    setTabIndex(index);
  };
  return (
    <div className="sprucemarkdown_tabs mt_60 flex column align_center text_center justify_center gap_24">
      <h3>Neat and Simple but Powerful</h3>
      <div className="sprucemarkdown_tabs_list">
        <ReachTabs index={tabIndex} onChange={handleTabsChange}>
          <TabList>
            <Tab>{"/ai> command,"}</Tab>
            <Tab>Headers,</Tab>
            <Tab>Images,</Tab>
            <Tab>Lists,</Tab>
            <Tab>Tables,</Tab>
            <Tab>Code Blocks,</Tab>
            <Tab>Syntex highlight,</Tab>
            <Tab>Task list,</Tab>
            <Tab>Mathematics,</Tab>
            <Tab>AI Menu</Tab>
          </TabList>
          <div className="wrapper_content mt_30">
            <TabPanels>
              <TabPanel>
                <TabImage image={Image1} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image2} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image3} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image4} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image5} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image6} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image7} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image8} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image9} />
              </TabPanel>
              <TabPanel>
                <TabImage image={Image10} />
              </TabPanel>
            </TabPanels>
          </div>
        </ReachTabs>
      </div>
    </div>
  );
}

export default Tabs;
